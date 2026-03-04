#!/usr/bin/env python3
"""
SAM 3 Text Prompt Helper

- Uses Meta's official `sam3` package and text prompts instead of box prompts
- Loads the same real data format as `sam_3_helper.py` and `sam_3_text_helper2.py`
- Text prompts are taken from YOLO labels in `yolo_object_dict`
- Provides simple benchmarking + optional visualization similar to `sam_3_helper.py`

Author: Robin Eshraghi (adapted)
Created: 12/22/25

Usage examples:
    python sam_3_text_helper.py --real --max 50 --viz
    python sam_3_text_helper.py --real --device cpu

Assumptions:
- Real data is stored under:
      logs/current_run_outputs/offline_outputs/image_data_dumps/image_data_*.pkl
- Each pickle has a dict of `image_data` entries with:
      'rotated_rgb_image' (H, W, 3) uint8
      'yolo_object_dict' with:
          'rotated_bbox_xyxy'
          'label'

Requirements:
    pip install git+https://github.com/facebookresearch/sam3.git
"""

from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import redirect_stdout
from io import StringIO
import argparse
import pickle
import time

import numpy as np
import cv2
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# CONFIGURATION
# ============================================================================

_BENCHMARK_ENABLED = False
_PROFILE_ENABLED = False


# ============================================================================
# PROFILING
# ============================================================================

def _profile(func):
    """Minimal profiling decorator (zero overhead when disabled)."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _PROFILE_ENABLED:
            return func(*args, **kwargs)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0  # ms
        print(f"[PROFILE] {func.__name__}: {elapsed:.3f}ms")
        return result

    return wrapper


def _create_empty_masks(num: int, shape: Tuple[int, int]) -> List[np.ndarray]:
    """Create list of empty boolean masks."""
    return [np.zeros(shape, dtype=bool) for _ in range(num)]


def _visualize_benchmark_result(
    image: np.ndarray,
    bboxes: List[np.ndarray],
    masks: List[np.ndarray],
    labels: List[str],
    output_path: Path,
    processing_time: float = 0.0,
    image_number: int = 0,
) -> None:
    """
    Visualization: overlay all SAM3 text-prompt masks in green,
    and draw YOLO bboxes in red with their labels.
    """
    vis = image.copy()

    # --- Overlay all masks in bright green + contours ---
    overlay_color = np.array((0, 255, 0), dtype=np.uint8)  # bright green

    for mask in masks:
        if mask is None:
            continue

        m = np.array(mask)

        # Ensure 2D
        if m.ndim > 2:
            m = m[0]
        elif m.ndim < 2:
            continue

        # Binarize mask (SAM3 can return float/probabilities)
        m_bin = (m.astype(np.float32) > 0.5).astype(np.uint8)
        if m_bin.shape != vis.shape[:2]:
            m_bin = cv2.resize(m_bin, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)

        if m_bin.sum() == 0:
            continue

        mask_bool = m_bin.astype(bool)

        # Strong overlay so it's clearly visible
        vis[mask_bool] = (vis[mask_bool] * 0.2 + overlay_color * 0.8).astype(np.uint8)

        # Also draw mask contour
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # --- Draw YOLO bboxes in red with labels ---
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        label = labels[i] if i < len(labels) else ""
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)  # red bbox
        if label:
            cv2.putText(
                vis,
                str(label),
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    # Compose title + legend
    H, W = vis.shape[:2]
    title_height = 30
    legend_height = 150
    vis_with_legend = np.ones((H + title_height + legend_height, W, 3), dtype=np.uint8) * 255
    vis_with_legend[title_height:title_height + H, :] = vis

    font_title = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "YOLO bboxes + SAM3 text-prompt masks"
    title_scale = 0.7
    title_thickness = 2
    (text_width, _), _ = cv2.getTextSize(title_text, font_title, title_scale, title_thickness)
    title_x = (W - text_width) // 2
    title_y = title_height - 5
    cv2.putText(vis_with_legend, title_text, (title_x, title_y), font_title, title_scale, (0, 0, 0), title_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    y_offset = title_height + H + 20

    # Legend: colors
    cv2.rectangle(vis_with_legend, (10, y_offset - 10), (30, y_offset + 5), (0, 150, 0), -1)
    cv2.putText(
        vis_with_legend,
        "Green: SAM3 text-prompt masks",
        (35, y_offset + 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    cv2.rectangle(vis_with_legend, (10, y_offset + 15), (30, y_offset + 30), (255, 0, 0), -1)
    cv2.putText(
        vis_with_legend,
        "Red: YOLO bboxes + labels",
        (35, y_offset + 30),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    if image_number > 0:
        image_num_text = f"Image #{image_number}"
        cv2.putText(
            vis_with_legend,
            image_num_text,
            (10, y_offset + 60),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    if processing_time > 0:
        time_text = f"Processing time: {processing_time * 1000:.2f}ms"
        cv2.putText(
            vis_with_legend,
            time_text,
            (10, y_offset + 80),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    cv2.imwrite(str(output_path), cv2.cvtColor(vis_with_legend, cv2.COLOR_RGB2BGR))


# ============================================================================
# DATA LOADING (real data, YOLO labels as text prompts)
# ============================================================================

def load_real_text_cases(
    data_dir: Path,
    max_cases: int = 10,
) -> List[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    """
    Load real test cases from image_data_dumps.

    Returns list of tuples: (image, bboxes, labels)
        image: (H, W, 3) RGB uint8
        bboxes: list of [x1, y1, x2, y2]
        labels: list of str (YOLO labels), same length/order as bboxes
    """
    image_data_dir = data_dir / "image_data_dumps"
    if not image_data_dir.exists():
        print(f"[ERROR] image_data_dumps not found at: {image_data_dir}")
        return []

    test_cases: List[Tuple[np.ndarray, List[np.ndarray], List[str]]] = []

    for pkl_path in sorted(image_data_dir.glob("image_data_*.pkl")):
        with open(pkl_path, "rb") as f:
            viz_data = pickle.load(f)

        for image_data in viz_data.values():
            image = image_data.get("rotated_rgb_image")
            if image is None:
                continue

            yolo_dict = image_data.get("yolo_object_dict", {})
            bboxes: List[np.ndarray] = []
            labels: List[str] = []

            for obj in yolo_dict.values():
                bbox = obj.get("rotated_bbox_xyxy")
                label = obj.get("label", "")
                if bbox is None or label == "":
                    continue
                bboxes.append(np.array(bbox))
                labels.append(str(label))

            if bboxes:
                test_cases.append((image, bboxes, labels))
                if len(test_cases) >= max_cases:
                    return test_cases

    return test_cases


# ============================================================================
# CORE: SAM3 TEXT PROMPT INFERENCE
# ============================================================================

@_profile
def get_sam3_masks_from_text_prompts(
    processor: Sam3Processor,
    image: np.ndarray,
    text_prompts: List[str],
    device: torch.device,
) -> List[np.ndarray]:
    """
    Run SAM3 text-prompt segmentation using Meta's official API.

    - Sets image once, then runs one text prompt at a time (same as sam_3_text_helper2.py)
    - Returns one mask per text prompt (empty mask if prompt fails)
    """
    H, W = image.shape[:2]

    if not text_prompts:
        return []

    # Ensure PIL image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Set image once
    try:
        with redirect_stdout(StringIO()):
            inference_state = processor.set_image(pil_image)
    except Exception as e:
        print(f"[ERROR] set_image failed for this image: {e}")
        return _create_empty_masks(len(text_prompts), (H, W))

    masks: List[np.ndarray] = []

    for text_prompt in text_prompts:
        try:
            with redirect_stdout(StringIO()):
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompt,
                )

            masks_tensor = output.get("masks", None)
            scores = output.get("scores", None)

            if masks_tensor is None:
                masks.append(np.zeros((H, W), dtype=bool))
                continue

            # Move masks to CPU NumPy
            if isinstance(masks_tensor, torch.Tensor):
                masks_np = masks_tensor.detach().cpu().numpy()
            elif isinstance(masks_tensor, np.ndarray):
                masks_np = masks_tensor
            else:
                masks_np = np.array(masks_tensor)

            # Expect masks_np shape (N, Hm, Wm); if not, treat as no masks
            if masks_np.ndim != 3 or masks_np.shape[0] == 0:
                masks.append(np.zeros((H, W), dtype=bool))
                continue

            # Choose the best mask for this prompt (by score if available, otherwise first)
            if scores is not None:
                # Move scores to CPU NumPy and validate
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.detach().cpu().numpy()
                elif isinstance(scores, np.ndarray):
                    scores_np = scores
                else:
                    scores_np = np.array(scores)

                scores_flat = scores_np.reshape(-1)
                if scores_flat.size == 0:
                    best_idx = 0
                else:
                    best_idx = int(scores_flat.argmax())
            else:
                best_idx = 0

            # Safety: clamp index into [0, N-1]
            best_idx = max(0, min(best_idx, masks_np.shape[0] - 1))
            best_mask = masks_np[best_idx]

            # Ensure 2D mask
            if best_mask.ndim > 2:
                best_mask = best_mask[0]
            elif best_mask.ndim < 2:
                masks.append(np.zeros((H, W), dtype=bool))
                continue

            if best_mask.shape != (H, W):
                best_mask = cv2.resize(
                    best_mask.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                best_mask = best_mask.astype(bool)

            masks.append(best_mask)

        except Exception as e:
            print(f"[ERROR] Text prompt {repr(text_prompt)} failed: {e}")
            masks.append(np.zeros((H, W), dtype=bool))

    # Guarantee length
    while len(masks) < len(text_prompts):
        masks.append(np.zeros((H, W), dtype=bool))

    return masks[: len(text_prompts)]


# ============================================================================
# BENCHMARK / MAIN LOOP
# ============================================================================

def run_text_benchmark(
    max_cases: int = 20,
    enable_viz: bool = False,
    device_str: str = "cuda",
) -> None:
    """Run SAM3 text-prompt benchmark on real data with optional visualization."""
    global _BENCHMARK_ENABLED

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # vision_grounding -> tiamat_agent -> tiamatl_eval_mvp
    data_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs"

    print(f"[INFO] Data root: {data_dir}")

    test_cases = load_real_text_cases(data_dir, max_cases=max_cases)
    if not test_cases:
        print("[ERROR] No test cases loaded (no images with YOLO labels).")
        return
    print(f"[INFO] Loaded {len(test_cases)} image(s) with YOLO labels.\n")

    # Visualization directory
    viz_dir: Optional[Path] = None
    if enable_viz:
        viz_dir = (
            project_root
            / "logs"
            / "current_run_outputs"
            / "offline_outputs"
            / "sam_3_text_helper_benchmark"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Visualization enabled: saving to {viz_dir}\n")

    # Device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # Build Meta SAM3 image model + processor (same as sam_3_text_helper2.py)
    print("[INFO] Building SAM3 image model (Meta sam3 package)...")
    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model)
    print(f"[INFO] SAM3 image model ready on device={device}\n")

    _BENCHMARK_ENABLED = True

    times: List[float] = []
    total_objects = 0

    for idx, (image, bboxes, labels) in enumerate(test_cases):
        print(f"[INFO] Image {idx+1}/{len(test_cases)} with {len(bboxes)} objects")
        print(f"       Text prompts: {labels}")

        start = time.perf_counter()
        masks = get_sam3_masks_from_text_prompts(processor, image, labels, device=device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        num_bboxes = len(bboxes)
        total_objects += num_bboxes

        print(
            f"       Stats: current={num_bboxes}, masks={len(masks)}, "
            f"time={elapsed*1000:.2f}ms"
        )

        if enable_viz and viz_dir is not None:
            output_path = viz_dir / f"sam3_text_{idx:03d}.png"
            _visualize_benchmark_result(
                image,
                bboxes,
                masks,
                labels,
                output_path,
                processing_time=elapsed,
                image_number=idx + 1,
            )

        print()

    _BENCHMARK_ENABLED = False

    if times:
        times_ms = [t * 1000.0 for t in times]
        print("=" * 70)
        print(
            f"Results: {np.mean(times_ms):.2f}ms avg | "
            f"{np.min(times_ms):.2f}ms min | {np.max(times_ms):.2f}ms max"
        )
        print(f"FPS: {1000.0 / np.mean(times_ms):.2f}")
        print(f"Total objects: {total_objects}")
        print("=" * 70)


def main() -> None:
    global _PROFILE_ENABLED

    parser = argparse.ArgumentParser(
        description="SAM3 Text Prompt Helper using Meta's `sam3` package and YOLO labels as prompts.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real data from logs/current_run_outputs/offline_outputs (required).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization and save images to logs/.../sam_3_text_helper_benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable lightweight profiling prints.",
    )

    args = parser.parse_args()
    _PROFILE_ENABLED = args.profile

    if args.profile:
        print("[PROFILE] Profiling enabled\n")

    if not args.real:
        print(
            "ERROR: This helper currently only supports real data mode.\n"
            "Please re-run with `--real`.\n"
            "Example:\n"
            "    python sam_3_text_helper.py --real --max 50 --viz"
        )
        return

    run_text_benchmark(
        max_cases=args.max,
        enable_viz=args.viz,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()



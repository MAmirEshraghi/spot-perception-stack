#!/usr/bin/env python3
"""
SAM 3 Text Prompt Debug Helper (Simple, Meta sam3 package)

Goal:
- Load the same real data as `sam_3_text_helper.py`
- For each image:
    * collect YOLO labels -> text prompts
    * run Meta's official sam3 image model with text prompts
    * log: prompts and number of masks/boxes for debugging

Usage:
    python sam_3_text_helper2.py          # default: use real data (first N images)
    python sam_3_text_helper2.py --max 50 # limit number of images

Assumptions:
- Real data is stored under:
    logs/current_run_outputs/offline_outputs/image_data_dumps/image_data_*.pkl
- Each pickle has a dict of `image_data` entries with:
    - 'rotated_rgb_image'
    - 'yolo_object_dict' with fields:
        * 'rotated_bbox_xyxy'
        * 'label'

Requirements:
    pip install git+https://github.com/facebookresearch/sam3.git
"""

from pathlib import Path
import argparse
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_real_images_and_labels(data_root: Path, max_images: int) -> List[Tuple[Image.Image, List[str]]]:
    """
    Load (PIL_image, unique_labels) pairs from real data dumps.

    Each pickle file:
      - has a dict of image_data entries
      - each entry contains:
          'rotated_rgb_image' (H, W, 3) uint8
          'yolo_object_dict' with 'label'
    """
    image_data_dir = data_root / "image_data_dumps"
    if not image_data_dir.exists():
        print(f"[ERROR] image_data_dumps not found at: {image_data_dir}")
        return []

    pairs: List[Tuple[Image.Image, List[str]]] = []
    for pkl_path in sorted(image_data_dir.glob("image_data_*.pkl")):
        with open(pkl_path, "rb") as f:
            viz_data = pickle.load(f)

        for image_data in viz_data.values():
            rgb = image_data.get("rotated_rgb_image")
            if rgb is None:
                continue

            yolo_dict = image_data.get("yolo_object_dict", {})
            labels: List[str] = []
            for obj in yolo_dict.values():
                label = obj.get("label", "")
                if label:
                    labels.append(label)

            if not labels:
                continue

            unique_labels = sorted(set(labels))
            pil_image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
            pairs.append((pil_image, unique_labels))

            if len(pairs) >= max_images:
                return pairs

    return pairs


def run_simple_sam3_debug(
    max_images: int = 20,
    device: str = "cuda",
) -> None:
    """Run a simple per-image SAM 3 text-prompt debug loop using Meta's sam3 package."""
    # Resolve data root (same logic as main helper)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # vision_grounding -> tiamat_agent -> tiamatl_eval_mvp
    data_root = project_root / "logs" / "current_run_outputs" / "offline_outputs"

    print(f"[INFO] Data root: {data_root}")

    # Load images and labels
    pairs = load_real_images_and_labels(data_root, max_images=max_images)
    if not pairs:
        print("[ERROR] No images with labels found.")
        return
    print(f"[INFO] Loaded {len(pairs)} image(s) with labels.\n")

    # Initialize Meta SAM3 image model and processor
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[INFO] Building SAM3 image model (Meta sam3 package)...")
    model = build_sam3_image_model().to(device_t)
    processor = Sam3Processor(model)
    print(f"[INFO] SAM3 image model ready on device={device_t}\n")

    # Main debug loop
    for idx, (image, labels) in enumerate(pairs):
        print(f"Image {idx+1}/{len(pairs)}")
        print(f"  Text prompts ({len(labels)}): {labels}")

        # Ensure PIL image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Set image once for this batch of prompts
        try:
            inference_state = processor.set_image(pil_image)
        except Exception as e:
            print(f"  [ERROR] set_image failed for this image: {e}\n")
            continue

        for text_prompt in labels:
            try:
                # Run text prompt segmentation using Meta's API
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompt,
                )

                masks = output.get("masks", None)
                boxes = output.get("boxes", None)
                scores = output.get("scores", None)

                num_masks = 0 if masks is None else masks.shape[0]
                boxes_shape = None if boxes is None else tuple(boxes.shape)
                scores_shape = None if scores is None else tuple(scores.shape)

                print(
                    f"    Prompt: {repr(text_prompt)} -> "
                    f"{num_masks} mask(s), "
                    f"boxes={boxes_shape}, "
                    f"scores={scores_shape}"
                )
            except Exception as e:
                print(f"    [ERROR] Prompt {repr(text_prompt)} failed: {e}")

        print()

    print("[INFO] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Meta SAM3 text-prompt debug helper.")
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g., 'cuda' or 'cpu').",
    )
    args = parser.parse_args()

    run_simple_sam3_debug(max_images=args.max, device=args.device)


if __name__ == "__main__":
    main()


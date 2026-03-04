#!/usr/bin/env python3
"""
Object Candidate Selection Module

This module loads and visualizes object records for candidate selection.

Uses two-stage selection: embedding similarity + local LLM filtering.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiamat_agent.utils.plotters import plot_object_records_top_down_interactive
from tiamat_agent.utils.session_logger import SessionLogger
from tiamat_agent.vision_grounding.z_sensor_object_map_node import ObjectLibrary

class LocalLLM:
    """Minimal LLM wrapper."""
    
    def __init__(self, model_name: str = "qwen/Qwen3-4B-Instruct-2507", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LLM: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ LLM loaded")
    
    def complete(self, prompt: str, max_new_tokens: int = 200) -> str:
        start_time = time.time()
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        elapsed = time.time() - start_time
        print(f"  LLM completion: {elapsed:.2f}s")
        return response.strip()

import pickle as pk
def load_object_library(object_library_path: Path) -> ObjectLibrary:
    """
    Load object library from either pickle (.pkl) or JSON (.json) format.
    
    For JSON: Preserves original object_id values as dictionary keys.
    For Pickle: Returns ObjectLibrary as-is.
    
    Args:
        object_library_path: Path to .pkl (ObjectLibrary) or .json file
    
    Returns:
        ObjectLibrary instance with objects_by_id dict
    """
    object_library_path = Path(object_library_path)
        
    # Handle JSON format
    if object_library_path.suffix == '.json':
        print(f"[LOAD] Loading from JSON file: {object_library_path}")
        with open(object_library_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract objects list (handle both direct list and wrapped format)
        if isinstance(data, list):
            objects_list = data
        elif isinstance(data, dict) and "objects" in data:
            objects_list = data["objects"]
        else:
            raise ValueError(f"JSON format not recognized. Expected list or dict with 'objects' key")
        
        print(f"[LOAD] Found {len(objects_list)} objects in JSON file")
        
        # Create ObjectLibrary and preserve original IDs
        object_library = ObjectLibrary()
        id_types = {"int": 0, "str": 0, "none": 0}
        
        for obj in objects_list:
            original_id = obj.get("object_id")
            
            if original_id is not None:
                # Track ID types for debugging
                if isinstance(original_id, int):
                    id_types["int"] += 1
                elif isinstance(original_id, str):
                    id_types["str"] += 1
                
                # Preserve original ID as dictionary key (string or int)
                object_library.objects_by_id[original_id] = obj
            else:
                id_types["none"] += 1
                # Fallback: assign sequential ID if missing
                obj_id = len(object_library.objects_by_id)
                object_library.objects_by_id[obj_id] = obj
                object_library.objects_by_id[obj_id]["object_id"] = obj_id
        
        # Show sample of IDs
        sample_ids = list(object_library.objects_by_id.keys())[:5]
        print(f"[LOAD] ID types in JSON: {id_types}")
        print(f"[LOAD] Sample IDs (first 5): {sample_ids}")
        print(f"[LOAD] Successfully loaded {len(object_library)} objects from JSON")
        
        return object_library
    
    # Handle pickle format (default/fallback)
    else:
        print(f"[LOAD] Loading from pickle file: {object_library_path}")
        with open(object_library_path, 'rb') as f:
            object_library = pk.load(f)
        
        # Validate it's an ObjectLibrary instance
        if not isinstance(object_library, ObjectLibrary):
            raise ValueError(
                f"Pickle file must contain ObjectLibrary instance, "
                f"got {type(object_library).__name__}"
            )
        
        print(f"[LOAD] Successfully loaded {len(object_library)} objects from pickle")
        return object_library


class ObjectCandidateSelector:
    """Selects relevant objects from a list of object records based on task description."""

    def __init__(self, session_id: str = "object_selection", llm: LocalLLM = None):
        self.session_logger = SessionLogger(session_id, "object_candidate_selection")
        self.logger = self.session_logger.get_logger()
        self.session_id = session_id
        self._embedder: Optional[SentenceTransformer] = None
        self._llm: Optional[LocalLLM] = llm

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedder is None:
            # Using all-mpnet-base-v2 for better semantic similarity (better quality than MiniLM)
            # Alternative: 'all-MiniLM-L12-v2' for faster but still good quality
            # self._embedder = SentenceTransformer('all-mpnet-base-v2')
            self._embedder = SentenceTransformer("clip-ViT-L-14")

        return self._embedder
    
    @property
    def llm(self) -> LocalLLM:
        """Lazy-load LLM."""
        if self._llm is None:
            self._llm = LocalLLM()
        return self._llm

    def _cache_label_embeddings(self, object_library: List[Dict]) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
        """Pre-compute embeddings for labels and object descriptions."""
        start_time = time.time()
        
        # Cache label embeddings
        labels = list(set(
            r["semantic_metadata"]["label"] for r in object_library.objects_by_id.values()
        ))
        label_embeddings = {}
        if labels:
            label_start = time.time()
            embeddings = self.embedder.encode(labels, convert_to_numpy=True)
            label_embeddings = dict(zip(labels, embeddings))
            print(f"  Label embeddings ({len(labels)} labels): {time.time() - label_start:.2f}s")
        
        end_time = time.time()
        self.logger.info(f"Label embeddings time: {end_time - start_time:.2f}s")

        start_time = time.time()
        # Cache description embeddings for each object
        description_embeddings = {}
        descriptions_to_encode = []
        obj_ids_for_descriptions = []
        
        for obj_id, obj in object_library.objects_by_id.items():
            description = obj.get("semantic_metadata", {}).get("description", "")
            if description:
                descriptions_to_encode.append(description)
                obj_ids_for_descriptions.append(obj_id)
        
        if descriptions_to_encode:
            desc_start = time.time()
            desc_embeddings = self.embedder.encode(descriptions_to_encode, convert_to_numpy=True)
            description_embeddings = dict(zip(obj_ids_for_descriptions, desc_embeddings))
            print(f"  Description embeddings ({len(descriptions_to_encode)} descriptions): {time.time() - desc_start:.2f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"Cached Description embeddings for {len(labels)} unique labels and {len(description_embeddings)} object descriptions (total: {total_time:.2f}s)")
        return label_embeddings, description_embeddings


    def _get_candidate_labels(
        self, 
        task_description: str,
        label_embeddings: Dict[str, np.ndarray],
        description_embeddings: Dict[int, np.ndarray],
        object_library: Dict,
        top_k: int = 20,
        threshold: float = 0.2
    ) -> List[tuple]:
        """
        Stage 1: Get candidate labels via embedding similarity.
        Embeds task description and compares with both labels and descriptions.
        For each label, takes the max similarity from label match or any object description match.
        """
        if not label_embeddings and not description_embeddings:
            return []
        
        start_time = time.time()
        
        # Embed the task description directly
        self.logger.info(f"Computing similarity using task description: '{task_description}'")
        task_emb_start = time.time()
        task_emb = self.embedder.encode(task_description, convert_to_numpy=True)
        print(f"  Task embedding: {time.time() - task_emb_start:.2f}s")
        
        # For each label, compute similarity to label and to all object descriptions with that label
        label_scores = {}
        
        # First, compute label similarities
        sim_start = time.time()
        for label, label_emb in label_embeddings.items():
            score = np.dot(task_emb, label_emb) / (
                np.linalg.norm(task_emb) * np.linalg.norm(label_emb)
            )
            label_scores[label] = float(score)
        
        # Then, compute description similarities and take max with label similarity
        for obj_id, obj in object_library.objects_by_id.items():
            label = obj["semantic_metadata"]["label"]
            if obj_id in description_embeddings:
                desc_emb = description_embeddings[obj_id]
                desc_score = np.dot(task_emb, desc_emb) / (
                    np.linalg.norm(task_emb) * np.linalg.norm(desc_emb)
                )
                # Update label score to be max of label similarity and this description similarity
                # label_scores[label] = max(label_scores.get(label, 0), float(desc_score))
                label_scores[label] = label_scores.get(label, 0)
        
        print(f"  Similarity computation: {time.time() - sim_start:.2f}s")
        
        # Filter by threshold and sort
        scored_labels = [
            (label, score) for label, score in label_scores.items() 
            if score >= threshold
        ]
        scored_labels.sort(key=lambda x: -x[1])
        result = scored_labels[:top_k]
        
        total_time = time.time() - start_time
        print(f"  Total embedding similarity stage: {total_time:.2f}s")
        return result
    
    def _llm_filter_labels(
        self, 
        task_description: str, 
        candidate_labels: List[str]
    ) -> List[str]:
        """Stage 2: LLM selects minimal subset of labels relevant to task."""
        if not candidate_labels:
            return []
        
        start_time = time.time()
        prompt = f"""Task: "{task_description}"

From the candidate labels below, select ONLY the object categories that are directly mentioned in the task or are essential to complete it.

Rules:
- If the task asks for a specific object, select only that object type
- Be minimal: select 1-6 objects maximum
- Only select objects that are explicitly needed for the task
- If no relevant objects are found, return an empty list []

Candidate labels: {candidate_labels}

Return ONLY a JSON array of selected labels, nothing else. """

         
        try:
            llm_start = time.time()
            response = self.llm.complete(prompt, max_new_tokens=200)
            print(f"  LLM filtering: {time.time() - llm_start:.2f}s")
            self.logger.info(f"LLM response: {response[:200]}")
            print(f"Task description: {task_description}, LLM response: {response}")
            
            # Extract JSON array from response
            response_clean = response.strip()
            
            # Remove markdown code blocks
            if "```" in response_clean:
                for part in response_clean.split("```"):
                    if "[" in part and "]" in part:
                        response_clean = part.strip()
                        if response_clean.startswith("json"):
                            response_clean = response_clean[4:].strip()
                        break
            
            # Find last JSON array
            start = response_clean.rfind("[")
            end = response_clean.rfind("]") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON array found")
            
            json_str = response_clean[start:end]
            selected = json.loads(json_str)
            
            # Validate: only return labels that exist in candidates
            valid_labels = [l for l in selected if l in candidate_labels]
            total_time = time.time() - start_time
            self.logger.info(f"LLM selected {len(valid_labels)}/{len(candidate_labels)} labels: {valid_labels} (total: {total_time:.2f}s)")
            print(f"  Total LLM filtering stage: {total_time:.2f}s")
            return valid_labels
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"LLM filtering failed: {e}, returning top 5 candidates (time: {total_time:.2f}s)")
            return candidate_labels[:5]

    def find_relevant_objects(
        self, 
        object_library: Dict,
        task_description: str, 
        top_k: int = 20,
        threshold: float = 0.2,
        use_llm: bool = True
    ) -> Tuple[List[Tuple[str, float]], List[Dict]]:
        """
        Find objects relevant to task using embedding similarity + LLM filtering.

        Args:
            object_records: List of object records to search through
            task_description: Natural language task
            top_k: Max candidates for embedding stage
            threshold: Minimum similarity score
            use_llm: Whether to use LLM for filtering

        Returns:
            Tuple of (candidate_labels, filtered_objects):
            - candidate_labels: List of (label, score) tuples from embedding similarity
            - filtered_objects: List of best object records after LLM filtering
        """
        total_start = time.time()
        print(f"\n=== Object Selection Timing ===")
        
        # Cache label and description embeddings from object records
        label_embeddings, description_embeddings = self._cache_label_embeddings(object_library)
        
        # Stage 1: Embedding similarity (task description vs labels and descriptions)
        print(f"Stage 1: Embedding similarity")
        candidate_labels = self._get_candidate_labels(
            task_description, 
            label_embeddings,
            description_embeddings,
            object_library,
            top_k, 
            threshold
        )
        
        if not candidate_labels:
            self.logger.warning("No candidate labels found")
            return [], []
        
        self.logger.info(f"Stage 1: {len(candidate_labels)} candidates from embedding")
        self.logger.info(f"Candidates (embedding similarity): {[f'{label} ({score:.3f})' for label, score in candidate_labels]}")
        
        # Stage 2: LLM filtering (optional, can be skipped if task_objects already filtered)
        labels_only = [label for label, _ in candidate_labels]
        if use_llm:
            print(f"Stage 2: LLM filtering")
            selected_labels = self._llm_filter_labels(task_description, labels_only)
        else:
            selected_labels = labels_only[:10]  # Fallback: top 10 by similarity
        
        self.logger.info(f"Stage 2: {len(selected_labels)} labels after LLM filtering")
        self.logger.info(f"Filtered labels (LLM selected): {selected_labels}")
        
        # Build score lookup
        score_lookup = dict(candidate_labels)
        
        # Group objects by label
        label_to_objects = defaultdict(list)
        for obj in object_library.objects_by_id.values():
            label_to_objects[obj["semantic_metadata"]["label"]].append(obj)
        
        # Return best object per selected label (highest confidence)
        results = []
        for label in selected_labels:
            objs = label_to_objects.get(label, [])
            if objs:
                best = max(objs, key=lambda o: o["detection_metadata"].get("yolo_score", 0))
                best["_relevance_score"] = score_lookup.get(label, 0)
                results.append(best)
        
        total_time = time.time() - total_start
        print(f"=== Total time: {total_time:.2f}s ===\n")
        self.logger.info(f"Total object selection time: {total_time:.2f}s")
        
        return candidate_labels, results

    def visualize_objects(self, object_records: List[Dict], world_map=None, show: bool = True,
                          save_path: Optional[Path] = None,
                          overlay_map: bool = False,
                          plot_robot: bool = True):
        """Visualize object records using plot_object_records_top_down."""
        # fig, ax = plt.subplots(figsize=(12, 12))
        plot_object_records_top_down_interactive(
        object_records,
        world_map, 
        overlay_map=True, 
        plot_robot=True,
        blacklist_labels=["floor", "wall", "ceiling", "cabinet", "couch", "carpet"],
        show=show#True
        )
        # plot_object_records_top_down_interactive(
        #     ax=ax,
        #     object_records=object_records,
        #     world_map=world_map,
        #     show=False,
        #     save_path=str(save_path) if save_path else None,
        #     overlay_map=overlay_map,
        #     plot_robot=plot_robot
        # )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

def load_saved_world_map(sample=False):
    """Load world map from current_run_outputs/map_dumps folder."""
    from pathlib import Path
    import glob
    import random
    from tiamat_agent.utils.session_logger import LOGS_FOLDER
    
    #map_path = Path("../data/saved_maps/world_map_dec_3.pkl")
    map_path = Path("tiamat_agent/data/saved_maps/world_map_dec_3.pkl")
    print(f"Loading world map from: {map_path}")
    with open(map_path, "rb") as f:
        world_map = pk.load(f)
    return world_map

def save_minimal_json_output(selector: ObjectCandidateSelector, task_results: Dict, test_tasks: Dict, args, object_library) -> Path:
    """
    Save minimal JSON output with essential fields for offline processing.
    
    Args:
        selector: ObjectCandidateSelector instance
        task_results: Dictionary mapping task_id to {"relevant_objects": [...], "candidate_labels": [...]}
        test_tasks: Dictionary of test tasks with prompts
        args: Command-line arguments
        object_library: ObjectLibrary instance
    
    Returns:
        Path to the saved JSON file
    """
    output_data = {
        "metadata": {
            "session_id": selector.session_id,
            "object_library_path": str(args.object_library_path),
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "tasks": []
    }
    
    for task_id, task_data in test_tasks.items():
        task_result = {
            "task_id": task_id,
            "task_prompt": task_data['prompt'],
            "selected_objects": []
        }
        
        # Add selected objects with minimal fields
        relevant_objects = task_results.get(task_id, {}).get("relevant_objects", [])
        for obj in relevant_objects:
            object_id = obj.get("object_id")
            if object_id is not None:
                # Get position_3d, handling different possible structures
                position_3d = [0.0, 0.0, 0.0]
                if "spatial_metadata" in obj and "position_3d" in obj["spatial_metadata"]:
                    pos = obj["spatial_metadata"]["position_3d"]
                    if isinstance(pos, (list, tuple, np.ndarray)):
                        position_3d = [float(x) for x in pos[:3]]
                    elif hasattr(pos, '__iter__'):
                        position_3d = [float(x) for x in list(pos)[:3]]
                
                # Extract frame_id and bbox_2d for visualization matching
                frame_id = None
                bbox_2d = None
                if "frame_metadata" in obj:
                    frame_id = obj["frame_metadata"].get("frame_id")
                if "detection_metadata" in obj:
                    bbox_2d = obj["detection_metadata"].get("bbox_2d")
                
                task_result["selected_objects"].append({
                    "object_id": int(object_id) if isinstance(object_id, (int, np.integer)) else str(object_id),
                    "label": obj["semantic_metadata"]["label"],
                    "relevance_score": float(obj.get("_relevance_score", 0.0)),
                    "position_3d": position_3d,
                    "frame_id": frame_id,
                    "bbox_2d": bbox_2d
                })
        
        output_data["tasks"].append(task_result)
    
    # Save to JSON file in logs/current_run_outputs/offline_outputs
    from tiamat_agent.utils.session_logger import LOGS_FOLDER
    output_dir = LOGS_FOLDER / "current_run_outputs" / "offline_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "object_selection_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Saved minimal JSON results to: {output_path}")
    print(f"{'='*70}\n")
    selector.logger.info(f"Saved minimal JSON results to: {output_path}")
    
    return output_path

def main():
    """Main function for testing."""
    import argparse
    # from tiamat_agent.utils.plotters import load_saved_world_map_from_current_run

    parser = argparse.ArgumentParser(description="Object candidate selection")
    parser.add_argument("--object-library-path", type=str,
                        default="../data/saved_maps/object_library_seed_data_dec_3.pkl")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    selector = ObjectCandidateSelector()
    print(f"\n{'='*70}")
    print(f"Loading object library from: {args.object_library_path}")
    print(f"{'='*70}")
    object_library = load_object_library(Path(args.object_library_path))
    selector.logger.info(f"Loaded {len(object_library)} objects from object library")

    # Test tasks
    test_tasks = {
        "1": {
            "prompt": "Dr. Strange will want a cup of coffee to start the day. Please help them find coffee and a mug.",
            "checklist": ["brewhouse_coffee_bag", "coffee_paper_bag", "mug"]
        },
        "2": {
            "prompt": "Find the map in the living room that shows where Dr Strange will be hiking.",
            "checklist": ["old_map"]
        },
        "3": {
            "prompt": "Gather items for hiking.",
            "checklist": ["backpack", "hiking_boots", "hiking_pole", "water_bottle", "sunscreen_bottle", "sun_hat", "headlamp"]
        },
        "4": {
            "prompt": "Dr Strange forgot their reading glasses on the nightstand, go upstairs and find them.",
            "checklist": ["reading_glasses"]
        },
        "5": {
            "prompt": "Dr Strange is testing a new ground sensor (which looks like a short white pole with a spherical white light at the top). When the sensor is active it will remain standing when an agent tries to push it over. If it can be pushed over it means the sensor is inactive. Find the ONE active sensor for Dr. Strange",
            "checklist": ["barbers_pole"]
        },
        "6": {
            "prompt": "Dr Strange would like a brownie for their lunch.",
            "checklist": ["plate_of_brownies"]
        },
        "7": {
            "prompt": "Dr. Strange's hiking boots are in a 'jumble' of shoes in the entry. Help them find the boots.",
            "checklist": ["dirty_boots", "arnt_shoes", "leather_shoes", "sneakers"]
        }
    }

    # Track object_ids selected for each task
    task_object_mapping = {}  # List of dicts: {"object_id": int, "task_id": str, "task_prompt": str}
    task_results = {}  # Store full results for JSON output: {task_id: {"relevant_objects": [...], "candidate_labels": [...]}}
    
    for task_id, task_data in test_tasks.items():
        task_object_mapping[task_id] = []
        print(f"\n{'='*70}")
        print(f"Task: {task_id}")
        print(f"Task: {task_data['prompt']}")
        print(f"Expected: {task_data['checklist']}")
        print(f"{'='*70}")
        
        candidate_labels, relevant = selector.find_relevant_objects(object_library, task_data['prompt'], top_k=20, use_llm=not args.no_llm)

        # Store results for JSON output
        task_results[task_id] = {
            "relevant_objects": relevant,
            "candidate_labels": candidate_labels
        }

        print(f"\n{'-'*70}")
        print(f"CANDIDATES FROM EMBEDDING SIMILARITY ({len(candidate_labels)} total):")
        print(f"{'-'*70}")
        for i, (label, score) in enumerate(candidate_labels, 1):
            print(f"  {i:2d}. {label:30s} (similarity={score:.3f})")
        
        print(f"\n{'-'*70}")
        print(f"FILTERED BY LLM ({len(relevant)} objects):")
        print(f"{'-'*70}")
        for i, obj in enumerate(relevant, 1):
            label = obj["semantic_metadata"]["label"]
            score = obj.get("_relevance_score", 0)
            conf = obj["detection_metadata"].get("yolo_score", 0)
            object_id = obj.get("object_id", None)
            print(f"  {i:2d}. {label:30s} (sim={score:.3f}, conf={conf:.3f}, object_id={object_id})")
            
            # Track this object_id for this task
            if object_id is not None:
                task_object_mapping[task_id].append({
                    "object_id": object_id,
                    "task_id": task_id,
                    "task_prompt": task_data['prompt']
                })
        print(f"{'='*70}")

    # At line 379: Output list of object_ids and their tasks
    print(f"\n{'='*70}")
    print(f"OBJECT ID TO TASK MAPPING ({len(task_object_mapping)} selections):")
    print(f"{'='*70}")
    for task_id, entries in task_object_mapping.items():
        print(f"{'='*70}")
        if len(entries) == 0:
            print(f"  No objects found for task {task_id}, Prompt: {test_tasks[task_id]['prompt']}")
            print("")
        else:
            print(f"  Task {task_id}: {test_tasks[task_id]['prompt']}")
            for entry in entries:
                object_id = entry['object_id']
                object_record = object_library.objects_by_id[object_id]
                label = object_record["semantic_metadata"]["label"]
                print(f"label={label}")
    print(f"{'='*70}\n")
    
    world_map = load_saved_world_map(sample=False)

    to_viz_object_records = []
    for task_id, entries in task_object_mapping.items():
        for entry in entries:
            object_id = entry['object_id']
            object_record = object_library.objects_by_id[object_id]
            object_record["semantic_metadata"]["label"] = object_record["semantic_metadata"]["label"] + " Task Id " + entry['task_id'] 
            object_record["task_description"] = entry['task_prompt']
            to_viz_object_records.append(object_record)
    print(f"To visualize {len(to_viz_object_records)} object records")
    selector.visualize_objects(to_viz_object_records, world_map=world_map, show=not args.no_show, save_path=args.save_plot, overlay_map=False, plot_robot=True)
    # selector.visualize_objects(object_library.objects_by_id.values(), world_map=world_map, show=not args.no_show, save_path=args.save_plot, overlay_map=False, plot_robot=True)
    
    # Save minimal JSON output
    save_minimal_json_output(selector, task_results, test_tasks, args, object_library)

if __name__ == "__main__":
    main()

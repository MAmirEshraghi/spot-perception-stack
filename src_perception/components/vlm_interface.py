#!/usr/bin/env python3
"""
VLM Interface for Object Detection
Provides a clean interface for different VLM implementations
"""

import json
import json_repair
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Dict
from PIL import Image
from abc import ABC, abstractmethod
from torchvision.transforms.functional import InterpolationMode
import flash_attn

class VLMInterface(ABC):
    """Abstract base class for VLM implementations"""
    
    @abstractmethod
    def detect_objects(self, images: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Detect objects in multiple images
        
        Args:
            images: Dict of {camera_name: rgb_image_array}
        
        Returns:
            Dict of {camera_name: list of object names found in that camera}
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name/identifier"""
        pass



# ============================================================================
# VLM Detector Implementations
# ============================================================================

class InternVLMDetector(VLMInterface):
    """InternVLM implementation for object detection"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3_5-1B", verbose: bool = True):
        """
        Initialize InternVLM detector
        
        Args:
            model_name: HuggingFace model identifier
            verbose: Print loading and processing info
        """
        self.model_name = model_name
        self.verbose = verbose
        self.pipe = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the VLM pipeline"""
        if self.verbose:
            print(f"\n🤖 Loading VLM pipeline: {self.model_name}")
        
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig
            from multiprocessing import freeze_support
            freeze_support()
            
            self.pipe = pipeline(
                self.model_name, 
                backend_config=TurbomindEngineConfig(session_len=8196, tp=1)
            )
            
            if self.verbose:
                print("   VLM pipeline loaded successfully")
                
        except Exception as e:
            print(f"   Failed to create VLM pipeline: {e}")
            raise
    
    def detect_objects(self, images: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Detect objects across multiple camera images
        
        Args:
            images: Dict of {camera_name: rgb_image_array (H, W, 3) uint8}
        
        Returns:
            Dict of {camera_name: sorted list of object names found in that camera}
        """
        if not images:
            return {}
        
        # Convert numpy images to PIL format
        pil_images = {}
        for camera_name, rgb_image in images.items():
            try:
                pil_image = self._numpy_to_pil(rgb_image)
                pil_images[camera_name] = pil_image
            except Exception as e:
                if self.verbose:
                    print(f"      Error converting {camera_name}: {e}")
                continue
        
        if not pil_images:
            return {}
        
        # Prepare batch prompts
        prompt_text = "Generate a JSON list of all objects in the image. Each object in the list must have exactly two keys: 'object_name' and 'description'."
        prompts = [(prompt_text, img) for img in pil_images.values()]
        camera_names = list(pil_images.keys())
        
        # Run VLM inference
        try:
            responses = self.pipe(prompts)
        except Exception as e:
            if self.verbose:
                print(f"      VLM inference failed: {e}")
            return {}
        
        # Parse responses and collect object names per camera
        objects_by_camera = {}
        
        for i, camera_name in enumerate(camera_names):
            camera_objects = set()
            try:
                response_text = json_repair.repair_json(responses[i].text)
                
                if self.verbose:
                    # Print truncated response for logging
                    truncated = response_text[:200] + "..." if len(response_text) > 200 else response_text
                    print(f"      {camera_name} raw response: {truncated}")
                
                # Parse JSON response
                detected_objects = json.loads(response_text)
                
                if isinstance(detected_objects, list):
                    for obj in detected_objects:
                        if isinstance(obj, dict) and "object_name" in obj:
                            # Clean and normalize object name
                            obj_name = str(obj["object_name"]).strip().lower()
                            if obj_name:
                                camera_objects.add(obj_name)
                    
                    if self.verbose:
                        print(f"      {camera_name}: Found {len(camera_objects)} objects")
                
            except Exception as e:
                if self.verbose:
                    print(f"      {camera_name}: Error parsing response - {e}")
            
            # Store sorted list for this camera
            objects_by_camera[camera_name] = sorted(list(camera_objects))
        
        return objects_by_camera
    
    def _numpy_to_pil(self, numpy_image: np.ndarray) -> Image.Image:
        """Convert numpy RGB image to PIL Image"""
        if numpy_image.dtype != np.uint8:
            numpy_image = numpy_image.astype(np.uint8)
        
        if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
            return Image.fromarray(numpy_image)
        else:
            raise ValueError(f"Unexpected image shape: {numpy_image.shape}")
    
    def get_model_name(self) -> str:
        """Return the model identifier"""
        return self.model_name


class InternVLMDetectorMulti(VLMInterface):
    """InternVLM implementation using multi-image query with transformers directly
    
    Simplified version: Resizes all images to input_size x input_size (no dynamic tiling).
    Best for consistent input image sizes like 720x720.
    Uses static preprocessing for fast batch processing.
    """
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3_5-1B", 
                 input_size: int = 448,
                 verbose: bool = True,
                 use_flash_attn: bool = True,
                 torch_compile: bool = False,
                 attn_implementation: str = "flash_attention_2"):
        """
        Initialize InternVLM detector with transformers backend
        
        Args:
            model_name: HuggingFace model identifier or local path
            input_size: Size to resize images to (default 448)
            verbose: Print loading and processing info
            use_flash_attn: Use flash attention for faster inference
        """
        self.model_name = model_name
        self.input_size = input_size
        self.verbose = verbose
        self.use_flash_attn = use_flash_attn
        self.torch_compile = torch_compile
        self.attn_implementation = attn_implementation
        
        self.model = None
        self.tokenizer = None
        self.transform = None
        
        # Load the model
        self._load_model()
        self._build_transform()
    
    def _load_model(self):
        """Load the VLM model and tokenizer"""
        if self.verbose:
            print(f"\n🤖 Loading InternVLM model: {self.model_name}")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                use_flash_attn=self.use_flash_attn,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            if self.verbose:
                device = next(self.model.parameters()).device
                print(f"   InternVLM model loaded successfully on {device}")
                
        except Exception as e:
            print(f"   Failed to load InternVLM model: {e}")
            raise
    
    def _build_transform(self):
        """Build image transformation pipeline for static preprocessing"""
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    
    def _static_preprocess(self, image: Image.Image) -> List[Image.Image]:
        """
        Static preprocessing - always produces 1 tile per image
        For consistent input sizes like 720x720
        """
        resized_img = image.resize((self.input_size, self.input_size))
        return [resized_img]
    
    def _preprocess_images(self, images: Dict[str, np.ndarray]) -> tuple:
        """
        Convert numpy images to preprocessed tensors
        
        Returns:
            pixel_values: Concatenated tensor of all images
            num_patches_list: List of patch counts per image
            camera_names: List of camera names in order
        """
        all_pixel_values = []
        num_patches_list = []
        camera_names = []
        
        for camera_name, rgb_image in images.items():
            try:
                # Convert numpy to PIL
                pil_image = self._numpy_to_pil(rgb_image)
                
                # Static preprocessing - produces 1 tile
                preprocessed_images = self._static_preprocess(pil_image)
                
                # Apply transforms
                pixel_values = [self.transform(img) for img in preprocessed_images]
                pixel_values = torch.stack(pixel_values)
                
                all_pixel_values.append(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                camera_names.append(camera_name)
                
            except Exception as e:
                if self.verbose:
                    print(f"      Error preprocessing {camera_name}: {e}")
                continue
        
        if not all_pixel_values:
            return None, None, None
        
        # Concatenate all pixel values
        pixel_values = torch.cat(all_pixel_values, dim=0)
        pixel_values = pixel_values.to(torch.bfloat16)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        return pixel_values, num_patches_list, camera_names
    
    def detect_objects(self, images: Dict[str, np.ndarray]) -> List[str]:
        """
        Detect objects across multiple camera images using multi-image query
        
        Args:
            images: Dict of {camera_name: rgb_image_array (H, W, 3) uint8}
        
        Returns:
            Sorted list of unique object names found across all images
        """
        if not images:
            return []
        
        # Preprocess all images
        pixel_values, num_patches_list, camera_names = self._preprocess_images(images)
        
        if pixel_values is None:
            return []
        
        # Build multi-image prompt
        num_images = len(camera_names)
        image_tags = "\n".join([f"Image-{i+1}: <image>" for i in range(num_images)])
        
        prompt = (
            f"{image_tags}\n"
            "Generate a JSON list of all objects across ALL images. "
            "Each object in the list must have exactly two keys: 'object_name' and 'description'."
        )
        
        # Generation configuration
        generation_config = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Run inference
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list
            )
            
            if self.verbose:
                truncated = response[:200] + "..." if len(response) > 200 else response
                print(f"      Multi-image VLM response: {truncated}")
            
            # Parse JSON response
            response_text = json_repair.repair_json(response)
            detected_objects = json.loads(response_text)
            
            # Extract object names
            all_object_names = set()
            
            if isinstance(detected_objects, list):
                for obj in detected_objects:
                    if isinstance(obj, dict) and "object_name" in obj:
                        obj_name = str(obj["object_name"]).strip().lower()
                        if obj_name:
                            all_object_names.add(obj_name)
                
                if self.verbose:
                    print(f"      Found {len(all_object_names)} unique objects across {num_images} images")
            
            return sorted(list(all_object_names))
            
        except Exception as e:
            if self.verbose:
                print(f"      Multi-image VLM inference failed: {e}")
            return []
    
    def _numpy_to_pil(self, numpy_image: np.ndarray) -> Image.Image:
        """Convert numpy RGB image to PIL Image"""
        if numpy_image.dtype != np.uint8:
            numpy_image = numpy_image.astype(np.uint8)
        
        if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
            return Image.fromarray(numpy_image)
        else:
            raise ValueError(f"Unexpected image shape: {numpy_image.shape}")
    
    def get_model_name(self) -> str:
        """Return the model identifier"""
        return f"{self.model_name}_multi"


# Factory function for easy instantiation
def create_vlm_detector(model_type: str = "internvlm", **kwargs) -> VLMInterface:
    """
    Factory function to create VLM detectors
    
    Args:
        model_type: Type of VLM ("internvlm", "internvlm_multi", etc.)
        **kwargs: Arguments passed to the specific detector
    
    Returns:
        VLMInterface instance
    
    Available types:
        - "internvlm": InternVLM with lmdeploy (per-camera inference)
        - "internvlm_multi": InternVLM with transformers (multi-image query)
    """
    if model_type.lower() == "internvlm":
        return InternVLMDetector(**kwargs)
    elif model_type.lower() == "internvlm_multi":
        return InternVLMDetectorMulti(**kwargs)
    else:
        raise ValueError(f"Unknown VLM type: {model_type}. Available: internvlm, internvlm_multi")


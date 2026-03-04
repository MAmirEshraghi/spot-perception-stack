#!/usr/bin/env python3

"""
Modifications (Robin):
    - change the return data fromat to:
        object_list = [{"object_name": name, "description": desc} for name, desc in camera_objects.items()]

    - add encode_text funcion for Script_2 (10/28/2025)

"""

"""
VLM Interface for Object Detection
Provides a clean interface for different VLM implementations
"""

import json
import json_repair
from matplotlib import image
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Dict
from PIL import Image
from abc import ABC, abstractmethod
from torchvision.transforms.functional import InterpolationMode
import traceback
VLM_ICON = "🤖"
class VLMInterface(ABC):
    """Abstract base class for VLM implementations"""
    
    @abstractmethod
    #def detect_objects(self, images: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    def detect_objects(self, images: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, str]]]:
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

    
    def _numpy_to_pil(self, numpy_image: np.ndarray) -> Image.Image:
        """Convert numpy RGB image to PIL Image"""
        if numpy_image.dtype != np.uint8:
            numpy_image = numpy_image.astype(np.uint8)
        
        if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
            return Image.fromarray(numpy_image)
        else:
            raise ValueError(f"Unexpected image shape: {numpy_image.shape}")
    



# ============================================================================
# VLM Detector Implementations
# ============================================================================

class InternVLMDetector(VLMInterface):
    """InternVLM implementation for object detection"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3_5-1B", logger= None):
        """
        Initialize InternVLM detector
        
        Args:
            model_name: HuggingFace model identifier
            verbose: Print loading and processing info
        """
        self.model_name = model_name
        self.logger = logger
        self.pipe = None
        self.vlm_icon = VLM_ICON
        self.vlm_prompt = "Generate a JSON list of all objects in the image. Each object in the list must have exactly two keys: 'object_name' and 'description'."
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the VLM pipeline"""
        self.logger.info(f"\n{self.vlm_icon} Loading VLM pipeline: {self.model_name}")
        
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig
            from multiprocessing import freeze_support
            freeze_support()
            
            self.pipe = pipeline(
                self.model_name, 
                backend_config=TurbomindEngineConfig(session_len=8196, tp=1)
            )
            
            self.logger.info(f"{self.vlm_icon} VLM pipeline loaded successfully")
        except Exception as e:
            self.logger.error(f"{self.vlm_icon} Failed to create VLM pipeline: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            raise
    
    def get_model_name(self) -> str:
        """Return the model identifier"""
        return self.model_name

    def detect_objects(self, images_by_id: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Detect objects across multiple camera images
        
        Args:
            images: Dict of {camera_name: rgb_image_array (H, W, 3) uint8}
        
        Returns:
            Dict of {camera_name: sorted list of object names found in that camera}
        """
        
        if not images_by_id:
            self.logger.warning(f"{self.vlm_icon} No images provided")
            return {}
        
        # Convert numpy images to PIL format
        image_bucket = {image_id : {"image": self._numpy_to_pil(rgb_image), "json_ouput":None} for image_id, rgb_image in images_by_id.items()}
        # Run VLM inference
        try:
            prompts = [(self.vlm_prompt, img["image"]) for img in image_bucket.values()]
            responses = self.pipe(prompts)
            for image_id, response in zip(image_bucket.keys(), responses):
                image_bucket[image_id]["json_output"] = json_repair.repair_json(response.text)
        except Exception as e:
            self.logger.error(f"{self.vlm_icon} VLM inference failed: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            return {}
        
        # Parse responses and collect object names per camera
        objects_by_id = {}
        for image_id, image_data in image_bucket.items():
            #camera_objects = set()
            object_list = []  # Use a dict to store name:description
            try:
                detected_objects = json.loads(image_data["json_output"])
                self.logger.info(f"{self.vlm_icon} {image_id} raw response: {image_data['json_output']}")
                # print(f"{self.vlm_icon} {image_id} raw response: {image_data['json_output'] [:200]}...")
                
                if isinstance(detected_objects, list):
                    for obj in detected_objects:
                        if isinstance(obj, dict) and "object_name" in obj:
                            # Clean and normalize object name
                            obj_name = str(obj["object_name"]).strip().lower()
                            if obj_name:
                                object_list.append({"object_name": obj_name, "description": obj.get("description", "")})
            except Exception as e:
                self.logger.error(f"{self.vlm_icon} {image_id}: Error parsing response - {e}")

            self.logger.info(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects")
            print(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects")

            objects_by_id[image_id] = object_list
        
        return objects_by_id


class QwenVLMDetector(VLMInterface):
    """Qwen2.5-VL implementation for object detection"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", logger=None, max_pixels: int = 1280 * 28 * 28, gen_description: bool = False):
        """
        Initialize Qwen2.5-VL detector
        
        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct", 
                       "Qwen/Qwen2-VL-7B-Instruct", or "Qwen/Qwen2-VL-2B-Instruct")
            logger: Logger instance
            max_pixels: Maximum pixels for image processing (default: 1280*28*28)
        """
        self.model_name = model_name
        self.logger = logger
        self.processor = None
        self.model = None
        self.device = None
        self.max_pixels = max_pixels
        self.vlm_icon = VLM_ICON
        self.gen_description = gen_description
        if self.gen_description:
            self.vlm_prompt = "Generate a JSON list of all objects in the image. Each object in the list must have exactly two keys: 'object_name' and 'description'."
        else:
            self.vlm_prompt = "Generate a JSON list of all objects in the image. Each object in the list must have exactly one key: 'object_name'."
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5-VL model using transformers - simplified version"""
        self.logger.info(f"\n{self.vlm_icon} Loading Qwen2.5-VL model: {self.model_name}")
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
            import torch
            
            # First check the model config to ensure it's actually a Qwen model
            try:
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                model_type = getattr(config, 'model_type', '').lower()
                
                # Check if this is actually a Qwen model
                if 'qwen' not in model_type and 'internvl' in model_type:
                    self.logger.error(f"{self.vlm_icon} Model '{self.model_name}' is an InternVL model, not Qwen!")
                    self.logger.error(f"{self.vlm_icon} Detected model_type: {model_type}")
                    self.logger.error(f"{self.vlm_icon} Please use a Qwen model name (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct') or change model_type to 'internvlm'")
                    raise ValueError(
                        f"Model type mismatch: '{self.model_name}' is an InternVL model (type: {model_type}), "
                        f"but Qwen2.5-VL detector was requested. "
                        f"Use a Qwen model name or set model_type='internvlm'."
                    )
                elif 'qwen' not in model_type:
                    self.logger.warning(f"{self.vlm_icon} Model type '{model_type}' may not be compatible with Qwen2.5-VL")
            except Exception as config_error:
                # If config loading fails, log warning but continue
                self.logger.warning(f"{self.vlm_icon} Could not verify model type: {config_error}")
            
            # Load processor first to check compatibility
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=256*28*28,
                max_pixels=self.max_pixels
            )
            
            # Verify processor is Qwen type
            processor_class_name = self.processor.__class__.__name__
            if "InternVL" in processor_class_name:
                self.logger.error(f"{self.vlm_icon} Processor mismatch: Loaded {processor_class_name} for Qwen model")
                self.logger.error(f"{self.vlm_icon} Model '{self.model_name}' appears to be InternVL, not Qwen")
                raise ValueError(
                    f"Processor mismatch: Expected Qwen processor but got {processor_class_name}. "
                    f"Model '{self.model_name}' is not a Qwen model. "
                    f"Use a Qwen model name (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct') or set model_type='internvlm'."
                )
            
            # Load model - simplest approach, no quantization
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Determine device for later use
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if torch.cuda.is_available():
                self.logger.info(f"{self.vlm_icon} Qwen2.5-VL model loaded successfully (CUDA)")
            else:
                self.logger.info(f"{self.vlm_icon} Qwen2.5-VL model loaded successfully (CPU)")
                
        except ValueError:
            # Re-raise our custom validation errors
            raise
        except Exception as e:
            # Check if error message indicates model type mismatch
            error_msg = str(e).lower()
            if 'internvl' in error_msg or 'qwen' in error_msg or 'model type' in error_msg:
                self.logger.error(f"{self.vlm_icon} Model type mismatch detected in error: {e}")
                self.logger.error(f"{self.vlm_icon} Model '{self.model_name}' may not be a Qwen model")
                raise ValueError(
                    f"Model type mismatch: '{self.model_name}' is not compatible with Qwen2.5-VL. "
                    f"Use a Qwen model name (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct') or set model_type='internvlm'."
                ) from e
            self.logger.error(f"{self.vlm_icon} Failed to load Qwen2.5-VL model: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            raise
    
    def get_model_name(self) -> str:
        """Return the model identifier"""
        return self.model_name

    def detect_objects(self, images_by_id: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, str]]]:
        """
        Detect objects across multiple camera images
        
        Args:
            images_by_id: Dict of {image_id: rgb_image_array (H, W, 3) uint8}
        
        Returns:
            Dict of {image_id: list of {"object_name": str, "description": str}}
        """
        
        if not images_by_id:
            self.logger.warning(f"{self.vlm_icon} No images provided")
            return {}
        
        # Convert numpy images to PIL format
        image_bucket = {image_id: {"image": self._numpy_to_pil(rgb_image), "json_output": None} 
                       for image_id, rgb_image in images_by_id.items()}
        
        # Run VLM inference for each image
        try:
            from qwen_vl_utils import process_vision_info
            import torch
            
            for image_id, image_data in image_bucket.items():
                try:
                    # Prepare inputs for Qwen2.5-VL - same format as template
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_data["image"]},
                                {"type": "text", "text": self.vlm_prompt}
                            ]
                        }
                    ]
                    
                    # Process inputs - exactly like the template
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    # Generate response - simple, no sampling
                    generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    
                    # Extract only the generated tokens (remove input tokens)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    # Decode the generated text
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    # Repair and store JSON output
                    image_data["json_output"] = json_repair.repair_json(output_text)
                    
                except Exception as e:
                    self.logger.error(f"{self.vlm_icon} {image_id}: VLM inference failed - {e}")
                    self.logger.error(traceback.format_exc())
                    image_data["json_output"] = "[]"
                    
        except ImportError:
            self.logger.error(f"{self.vlm_icon} qwen_vl_utils not installed. Please install it: pip install qwen-vl-utils")
            raise
        except Exception as e:
            self.logger.error(f"{self.vlm_icon} VLM inference failed: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            return {}
        
        # Parse responses and collect object names per camera
        objects_by_id = {}
        for image_id, image_data in image_bucket.items():
            object_list = []
            try:
                detected_objects = json.loads(image_data["json_output"])
                self.logger.info(f"{self.vlm_icon} {image_id} raw response: {image_data['json_output']}")
                
                if isinstance(detected_objects, list):
                    for obj in detected_objects:
                        if isinstance(obj, dict) and "object_name" in obj:
                            # Clean and normalize object name
                            obj_name = str(obj["object_name"]).strip().lower()
                            if obj_name:
                                object_list.append({"object_name": obj_name, "description": obj.get("description", "")})
            except Exception as e:
                self.logger.error(f"{self.vlm_icon} {image_id}: Error parsing response - {e}")

            self.logger.info(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects")
            print(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects")

            objects_by_id[image_id] = object_list
        
        return objects_by_id


class Qwen3VLMDetector(VLMInterface):
    """Qwen3-VL implementation for object detection with batch inference"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", logger=None, max_pixels: int = 1280 * 28 * 28, gen_description: bool = False):
        """
        Initialize Qwen3-VL detector with batch inference support
        
        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-VL-2B-Instruct")
            logger: Logger instance
            max_pixels: Maximum pixels for image processing (default: 1280*28*28)
            gen_description: Whether to generate descriptions for objects
        """
        self.model_name = model_name
        self.logger = logger
        self.processor = None
        self.model = None
        self.max_pixels = max_pixels
        self.vlm_icon = "🦉"  # Different icon to distinguish from Qwen2.5
        self.gen_description = gen_description
       # if self.gen_description:
        self.vlm_prompt = "Generate a JSON list of all objects in the image. Each object in the list must have exactly two keys: 'object_name' and 'description'. description should be 1-3 words."
        # else:
        #     self.vlm_prompt = "Generate a JSON list of all objects in the image. Each object in the list must have exactly one key: 'object_name'."
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen3-VL model using transformers with AutoModelForImageTextToText"""
        self.logger.info(f"\n{self.vlm_icon} Loading Qwen3-VL model: {self.model_name}")
        
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=256*28*28,
                max_pixels=self.max_pixels
            )
            
            # Load model with auto dtype and device mapping
            # For FP8 models, we need to handle quantization parameters properly
            # Try loading with low_cpu_mem_usage first, which helps with meta device handling
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except ValueError as e:
                # If meta device error occurs (common with FP8), try loading without device_map
                if "meta device" in str(e).lower():
                    self.logger.warning(f"{self.vlm_icon} Meta device error detected, trying alternative loading method...")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        torch_dtype="auto",
                        low_cpu_mem_usage=True
                    )
                    # Manually move to device if CUDA is available
                    if torch.cuda.is_available():
                        self.model = self.model.to("cuda")
                else:
                    raise
            
            if torch.cuda.is_available():
                self.logger.info(f"{self.vlm_icon} Qwen3-VL model loaded successfully (CUDA)")
            else:
                self.logger.info(f"{self.vlm_icon} Qwen3-VL model loaded successfully (CPU)")
                
        except Exception as e:
            self.logger.error(f"{self.vlm_icon} Failed to load Qwen3-VL model: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            raise
    
    def get_model_name(self) -> str:
        """Return the model identifier"""
        return self.model_name

    def detect_objects(self, images_by_id: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, str]]]:
        """
        Detect objects across multiple camera images using batch inference
        
        Args:
            images_by_id: Dict of {image_id: rgb_image_array (H, W, 3) uint8}
        
        Returns:
            Dict of {image_id: list of {"object_name": str, "description": str}}
        """
        
        if not images_by_id:
            self.logger.warning(f"{self.vlm_icon} No images provided")
            return {}
        
        # Convert numpy images to PIL format and track order
        image_ids = list(images_by_id.keys())
        image_bucket = {image_id: {"image": self._numpy_to_pil(rgb_image), "json_output": None} 
                       for image_id, rgb_image in images_by_id.items()}
        
        # Build batch of messages - one message list per image
        all_messages = []
        for image_id in image_ids:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_bucket[image_id]["image"]},
                    {"type": "text", "text": self.vlm_prompt}
                ]
            }]
            all_messages.append(messages)
        
        # Run batch VLM inference
        try:
            # Prepare inputs for batch processing
            inputs = self.processor.apply_chat_template(
                all_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True  # Required for batch inference!
            )
            inputs = inputs.to(self.model.device)
            
            # Single batch generation call with repetition penalty to avoid repeated objects
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.15)
            
            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode all outputs
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Assign outputs to corresponding image_ids
            for image_id, output_text in zip(image_ids, output_texts):
                image_bucket[image_id]["json_output"] = json_repair.repair_json(output_text)
                
        except Exception as e:
            self.logger.error(f"{self.vlm_icon} Batch VLM inference failed: {e}")
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            # Set empty outputs on failure
            for image_id in image_ids:
                image_bucket[image_id]["json_output"] = "[]"
        
        # Parse responses and collect object names per image
        objects_by_id = {}
        for image_id, image_data in image_bucket.items():
            object_list = []
            try:
                detected_objects = json.loads(image_data["json_output"])
                self.logger.info(f"{self.vlm_icon} {image_id} raw response: {image_data['json_output']}")
                
                if isinstance(detected_objects, list):
                    for obj in detected_objects:
                        if isinstance(obj, dict) and "object_name" in obj:
                            # Clean and normalize object name
                            obj_name = str(obj["object_name"]).strip().lower()
                            if obj_name:
                                object_list.append({"object_name": obj_name, "description": obj.get("description", "")})
            except Exception as e:
                self.logger.error(f"{self.vlm_icon} {image_id}: Error parsing response - {e}")

            self.logger.info(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects: {object_list}")
            print(f"{self.vlm_icon} {image_id}: Found {len(object_list)} objects: {object_list}")

            objects_by_id[image_id] = object_list
        
        return objects_by_id


# Factory function for easy instantiation
def create_vlm_detector(model_type: str = "internvlm", *args,  **kwargs) -> VLMInterface:
    """
    Factory function to create VLM detectors
    
    Args:
        model_type: Type of VLM ("internvlm", "qwen2.5", "qwen3", "qwen3-sgl", etc.)
        **kwargs: Arguments passed to the specific detector
    
    Returns:
        VLMInterface instance
    
    Available types:
        - "internvlm": InternVLM with lmdeploy (per-camera inference)
        - "qwen2.5" / "qwen2.5-vl": Qwen2.5-VL with transformers (per-image inference)
        - "qwen3" / "qwen3-vl": Qwen3-VL with batch inference (transformers)
    """
    model_type_lower = model_type.lower()
    if model_type_lower == "internvlm":
        return InternVLMDetector(*args, **kwargs)
    elif model_type_lower in ["qwen2.5", "qwen2.5-vl"]:
        # Set default model if not provided
        if "model_name" not in kwargs:
            kwargs["model_name"] = "Qwen/Qwen2.5-VL-3B-Instruct"
        return QwenVLMDetector(*args, **kwargs)
    elif model_type_lower in ["qwen3", "qwen3-vl"]:
        # Set default model if not provided
        if "model_name" not in kwargs:
            kwargs["model_name"] = "Qwen/Qwen3-VL-2B-Instruct"
        return Qwen3VLMDetector(*args, **kwargs)
    else:
        raise ValueError(f"Unknown VLM type: {model_type}. Available: internvlm, qwen2.5, qwen3")




if __name__ == "__main__":
    # Test Qwen3-VL with batch inference
    detector = create_vlm_detector(model_type="qwen3", model_name="Qwen/Qwen3-VL-2B-Instruct")
    detector.detect_objects(images_by_id={"image1": np.random.randint(0, 255, (1024, 1024, 3)).astype(np.uint8)})   
    print(detector.get_model_name())
    print(detector.detect_objects(images_by_id={"image1": np.random.randint(0, 255, (1024, 1024, 3)).astype(np.uint8)}))
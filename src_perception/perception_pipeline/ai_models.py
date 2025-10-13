from PIL import Image

# def get_vlm_description(cropped_object_image, model, processor):
#     """Generates a short description for a cropped image using a VLM."""
#     try:
#         pil_image = Image.fromarray(cropped_object_image).convert('RGB')  
#         messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "This object musked as a part of things inside of a home. Describe the object based on the observation and your creativety shortly, what is that?. If the image is not describable, you must respond with 'unknown object'."}]}]
#         prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to(model.device)
#         input_token_len = inputs["input_ids"].shape[1]
        
#         generated_ids = model.generate(**inputs, max_new_tokens=20)
#         description = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
#         return description.strip().lower()
#     except Exception as e:
#         print(f"[SmolVLM Error] {e}")
#         return "unknown object"
    
#     # In ai_models.py

# new func test:

def get_vlm_description(image_array, model, processor, prompt="Describe this image concisely."):
    """Generates a description for a given image using a flexible prompt."""
    try:
        
        pil_image = Image.fromarray(image_array).convert('RGB')
        
        messages = [{
            "role": "user", 
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": prompt} # Use the prompt argument here
            ]
        }]
        
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, images=[pil_image], return_tensors="pt").to(model.device)
        input_token_len = inputs["input_ids"].shape[1]
        
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        description = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
        
        return description.strip().lower()

    except Exception as e:
        print(f"[SmolVLM Error] {e}")
        return "unknown object"
    
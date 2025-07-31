import os
import base64

def generate_with_vllm(llm, sampling_params, conversation, image_paths=None):
    """Generate response using vLLM with support for images"""
    try:
        # For now, vLLM typically handles text-only models
        # Image support would depend on the specific model
        if image_paths:
            print("Warning: vLLM image support depends on the specific model being used")
            # For multimodal vLLM models, you would need to handle images differently
            # This is a placeholder for future multimodal support
        
        # Use llm.chat which handles the conversation template automatically
        outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        generated_tokens = len(outputs[0].outputs[0].token_ids)
        
        return {
            "text": generated_text,
            "token_ids": generated_tokens
        }
    except Exception as e:
        print(f"Error with vLLM: {e}")
        return {"text": "", "token_ids": 0}

def generate_with_vllm_multimodal(llm, sampling_params, conversation, image_paths=None):
    """Generate response using vLLM for multimodal models"""
    try:
        # For multimodal vLLM models like LLaVA
        # This is a more sophisticated handler for vision-language models
        
        # Build conversation
        messages = []
        for msg in conversation:
            if msg["role"] in ["system", "user", "assistant"]:
                messages.append(msg)
        
        # Handle images if present - this depends on the specific multimodal model
        if image_paths and any(os.path.exists(path) for path in image_paths if path):
            print(f"Processing {len(image_paths)} images with vLLM multimodal model")
            # For models like LLaVA, images are typically handled as part of the input
            # The exact implementation depends on the model's expected format
        
        # For now, fall back to text-only processing
        # Real multimodal support would require model-specific handling
        return generate_with_vllm(llm, sampling_params, conversation, image_paths)
        
    except Exception as e:
        print(f"Error with vLLM multimodal: {e}")
        return {"text": "", "token_ids": 0}
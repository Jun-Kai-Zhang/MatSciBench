import anthropic
import openai

import os
import base64
import json
import httpx
import mimetypes


from google import genai
from google.genai import types

def generate_with_api(model_type, model, conversation, max_tokens, temperature, image_paths=None):
    """Generate response using API with support for multiple images"""
    # Append datasets/ to the image paths if they don't already have it
    if image_paths:
        processed_image_paths = []
        for path in image_paths:
            # Only prepend datasets/ if it's not already there
            if path and not path.startswith("datasets/MatSciBench/"):
                processed_image_paths.append(f"datasets/MatSciBench/{path}")
            else:
                processed_image_paths.append(path)
        image_paths = processed_image_paths
    if model_type == "openai":
        response = generate_with_openai(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths instead of single path
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
    elif model_type == "deepseek":
        response = generate_with_deepseek(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
    elif model_type == "gemini":
        response = generate_with_gemini(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
        # If there was an error, include it in the output
        if response.get("error"):
            full_output = f"Error: {response['error']}\n\n{full_output}"
    elif model_type == "claude":
        response = generate_with_claude(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
    elif model_type == "qwen":
        response = generate_with_qwen(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
    elif model_type == "llama":
        response = generate_with_llama(
            model,
            conversation,
            max_tokens,
            temperature,
            image_paths  # Pass all image paths
        )
        full_output = response["text"].strip()
        new_token_nums = response["token_ids"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return full_output, new_token_nums


def generate_with_claude(model, conversation, max_tokens, temperature, image_paths=None):
    """Generate response using Claude API with support for multiple images"""
    try:
        # Convert conversation to Claude format
        messages = []
        
        # Extract system message if present
        system_content = ""
        for msg in conversation:
            if msg["role"] == "system":
                system_content = msg["content"]
                break
        
        # Format the conversation for Claude
        for i, msg in enumerate(conversation):
            if msg["role"] == "user":
                # For the last user message, check if we need to add images
                if i == len(conversation) - 1 and image_paths:
                    content = []
                    # Add the text content
                    content.append({
                        "type": "text",
                        "text": msg["content"]
                    })
                    
                    # Add images if they exist
                    for image_path in image_paths:
                        if os.path.exists(image_path):
                            try:
                                # Determine the correct mime type based on file extension
                                mime_type, _ = mimetypes.guess_type(image_path)
                                if not mime_type:
                                    mime_type = "image/jpeg"  # Default fallback
                                
                                with open(image_path, "rb") as image_file:
                                    image_data = image_file.read()
                                    content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime_type,
                                            "data": base64.b64encode(image_data).decode('utf-8')
                                        }
                                    })
                            except Exception as img_err:
                                print(f"Error loading image {image_path}: {img_err}")
                        else:
                            print("Image path does not exist: ", image_path)
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
            # Skip system messages as they're handled separately
        
        # Create Claude client and generate response
        client = anthropic.Anthropic(api_key=anthropic.api_key)
        response = client.messages.create(
            model=model,
            system=system_content,  # Pass system prompt here
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "text": response.content[0].text,
            "token_ids": response.usage.output_tokens
        }
    except Exception as e:
        print(f"Error with Claude API: {e}")
        return {"text": "", "token_ids": 0}
    


def generate_with_deepseek(
    model: str,
    conversation: list[dict],
    max_tokens: int,
    temperature: float | None,
    image_paths: list[str] | None = None,
):
    """
    Call DeepSeek models (chat or reasoner).  
    If model name contains 'reasoner', the function also extracts the
    chain-of-thought from `reasoning_content`.
    Returned dict:
        text          -- full text (CoT + final answer or just final answer)
        reasoning     -- CoT only (None for chat models)
        answer        -- assistant's final answer
        token_ids     -- completion tokens reported by the API
    """

    # ---------- 1. Build the message list, inserting images if supplied ----------
    messages = []
    for i, msg in enumerate(conversation):
        if (
            i == len(conversation) - 1
            and msg["role"] == "user"
            and image_paths
        ):
            content = [{"type": "text", "text": msg["content"]}]
            for p in image_paths:
                if os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        )
                    except Exception as img_err:
                        print(f"Image load failed for {p}: {img_err}")
            messages.append({"role": "user", "content": content})
        else:
            messages.append(msg)

    # ---------- 2. Prepare call arguments ----------
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    # deepseek-reasoner ignores sampling parameters, so only add them for chat models
    if temperature is not None and "reasoner" not in model:
        kwargs["temperature"] = temperature

    # ---------- 3. Call the API ----------
    try:
        response = openai.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # ---------- 4. Extract reasoning if present ----------
        reasoning = getattr(msg, "reasoning_content", None)
        answer = msg.content or ""
        if reasoning:
            full_text = f"{reasoning}\n\n---\n\n{answer}"
        else:
            full_text = answer

        return {
            "text": full_text,
            "token_ids": response.usage.completion_tokens,
        }

    except json.JSONDecodeError as e:
        error_msg = f"DeepSeek API returned invalid JSON: {e}"
        print(error_msg)
        
        # Try to extract any useful text from the raw response
        try:
            # For OpenAI client, we might not have direct access to the raw response
            # But we can try to get some information from the exception
            raw_response = str(e)
            
            # Look for text that might contain the model's response
            # Common patterns in responses even with JSON errors
            import re
            content_match = re.search(r'"content"\s*:\s*"([^"]+)"', raw_response)
            if content_match:
                extracted_text = content_match.group(1)
                return {"text": f"[Partial response from JSON error]: {extracted_text}", "token_ids": 0}
            
            # If nothing useful found, return the error message
            return {"text": f"JSON parsing error: {error_msg}", "token_ids": 0}
        except Exception as extract_err:
            print(f"Failed to extract content from error response: {extract_err}")
            return {"text": f"JSON parsing error: {error_msg}", "token_ids": 0}
            
    except httpx.HTTPStatusError as e:
        # HTTP errors might contain useful response data
        error_msg = f"DeepSeek API HTTP error: {e}"
        print(error_msg)
        
        # Try to extract any useful information from the response
        try:
            response_json = e.response.json()
            error_message = response_json.get("error", {}).get("message", "Unknown error")
            return {"text": f"API Error: {error_message}", "token_ids": 0}
        except Exception:
            # If we can't parse the JSON, try to get the raw text
            try:
                raw_text = e.response.text
                if len(raw_text) > 100:  # If there's substantial content
                    return {"text": f"API Error with response: {raw_text[:500]}...", "token_ids": 0}
            except Exception:
                pass
            return {"text": f"API Error: {str(e)}", "token_ids": 0}
            
    except Exception as e:
        error_msg = f"Unexpected error with DeepSeek API: {e}"
        print(error_msg)
        return {"text": error_msg, "token_ids": 0}
    

def generate_with_gemini(model, conversation, max_tokens, temperature, image_paths=None):
    """Generate response using Gemini API with support for multiple images"""
    try:
        # Create Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Find system prompt
        system_prompt = None
        for msg in conversation:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        # Format the conversation for Gemini
        contents = []
        
        # Process messages to create content history
        for i, msg in enumerate(conversation):
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                # For the first user message, prepend system prompt if available
                if not contents and system_prompt:
                    user_content = f"{system_prompt}\n\n{msg['content']}"
                else:
                    user_content = msg['content']
                    
                # For the last user message, handle images if present
                if i == len(conversation) - 1 and image_paths:
                    parts = [{"text": user_content}]
                    
                    # Add images if they exist
                    for image_path in image_paths:
                        if os.path.exists(image_path):
                            try:
                                with open(image_path, "rb") as image_file:
                                    image_data = image_file.read()
                                    parts.append({"inline_data": {"mime_type": "image/png", "data": image_data}})
                            except Exception as img_err:
                                print(f"Error loading image {image_path}: {img_err}")
                        else:
                            print("Image path does not exist: ", image_path)
                    
                    contents.append({"role": "user", "parts": parts})
                else:
                    contents.append({"role": "user", "parts": [{"text": user_content}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg['content']}]})
        
        # Configure generation parameters
        if model.startswith("gemini-2.5"):
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                # max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=max_tokens)
            )
        else:
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        # Generate content with the new API style
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config
        )
        
        # print("Usage metadata: ", response.usage_metadata)
        return {
            "text": response.text,
            "token_ids": response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count
        }
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return {"text": "", "token_ids": 0}
    

def generate_with_openai(model, conversation, max_tokens, temperature, image_paths=None):
    """Generate response using OpenAI API with support for multiple images"""
    try:
        # Format messages for potential image inclusion
        messages = []
        
        # Process all messages
        for i, msg in enumerate(conversation):
            # For the last user message, check if we need to add images
            if i == len(conversation) - 1 and msg["role"] == "user" and image_paths:
                # Create content array format for multimodal inputs
                content = [
                    {"type": "text", "text": msg["content"]}
                ]
                
                # Add images if they exist
                image_added = False
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        try:
                            # Get the image data as base64
                            with open(image_path, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                                
                                # Add image to content
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    }
                                })
                                image_added = True
                                # print("Add image to content: ", image_path)
                        except Exception as img_err:
                            print(f"Error loading image {image_path}: {img_err}")
                    else:
                        print("Image path does not exist: ", image_path)
                if image_added:
                    messages.append({"role": msg["role"], "content": content})
                else:
                    messages.append(msg)  # Fall back to text-only if no images worked
            else:
                messages.append(msg)
        
        # Make the API call - handle newer o3/o1 models differently
        if model.startswith("o4-") or model.startswith("o3-") or model.startswith("o1-"):
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,  # Use max_completion_tokens for o3/o1 models
            )
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,  # Use max_tokens for other models
                temperature=temperature
            )
        return {
            "text": response.choices[0].message.content,
            "token_ids": response.usage.completion_tokens
        }
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return {"text": "", "token_ids": 0}



def generate_with_qwen(
    model: str,
    conversation: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    image_paths: list[str] | None = None,
):
    """
    Call Qwen (DashScope) models, including reasoning models such as QwQ-32B.

    Returns a dict with keys
        text       – full text (reasoning + final answer, or just answer)
        reasoning  – chain of thought (None for plain chat models)
        answer     – assistant's final answer
        token_ids  – int, completion tokens reported by usage
    """
    import os, base64, openai

    # 1. Build message list and embed images if provided
    messages = []
    for i, msg in enumerate(conversation):
        if (
            i == len(conversation) - 1
            and msg["role"] == "user"
            and image_paths
        ):
            content = [{"type": "text", "text": msg["content"]}]
            for p in image_paths:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                else:
                    print("Image path does not exist: ", p)
            messages.append({"role": "user", "content": content})
        else:
            messages.append(msg)

    # 2. Create client bound to DashScope compatible endpoint
    client = openai.OpenAI(
        api_key=openai.api_key,
        base_url=openai.base_url,  # make sure this points to https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    )

    # 3. Decide whether we will receive reasoning traces
    wants_reasoning = "qwq" in model.lower()

    # 4. Stream the reply, asking to include usage
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )

    answer_buf, reasoning_buf = [], []
    usage = {"completion_tokens": 0}

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                answer_buf.append(delta.content)
            if wants_reasoning and getattr(delta, "reasoning_content", None):
                reasoning_buf.append(delta.reasoning_content)
        if chunk.usage:  # last chunk only
            usage = chunk.usage

    answer = "".join(answer_buf)
    reasoning = "".join(reasoning_buf) or None
    full_text = f"{reasoning}\n\n---\n\n{answer}" if reasoning else answer

    return {
        "text": full_text,
        "token_ids": usage.completion_tokens if hasattr(usage, "completion_tokens") else 0,
    }

def generate_with_llama(model, conversation, max_tokens, temperature, image_paths=None):
    """Generate response using Lambda Labs API for Llama models"""
    try:
        # Create OpenAI client with Lambda Labs API endpoint
        client = openai.OpenAI(
            api_key=openai.api_key,
            base_url="https://api.lambda.ai/v1"
        )
        
        # Format messages for the OpenAI-compatible API
        messages = []
        
        # Process all messages
        for i, msg in enumerate(conversation):
            # For the last user message, check if we need to add images
            if i == len(conversation) - 1 and msg["role"] == "user" and image_paths:
                # Create content array format for multimodal inputs
                content = [
                    {"type": "text", "text": msg["content"]}
                ]
                
                # Add images if they exist
                image_added = False
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        try:
                            # Get the image data as base64
                            with open(image_path, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                                
                                # Add image to content
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    }
                                })
                                image_added = True
                                # print("Add image to content: ", image_path)
                        except Exception as img_err:
                            print(f"Error loading image {image_path}: {img_err}")
                    else:
                        print("Image path does not exist: ", image_path)
                if image_added:
                    messages.append({"role": msg["role"], "content": content})
                else:
                    messages.append(msg)  # Fall back to text-only if no images worked
            else:
                messages.append(msg)
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "text": response.choices[0].message.content,
            "token_ids": response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0
        }
    except Exception as e:
        print(f"Error with Llama API: {e}")
        return {"text": "", "token_ids": 0}

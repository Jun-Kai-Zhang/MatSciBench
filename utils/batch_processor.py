import json
import os
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Tuple
import openai
import anthropic

def get_prepare_prompt_function(method_function):
    """Get the prepare_prompt function from the method's module."""
    # Get the module that contains the method function
    method_module = method_function.__module__

    # Import the module dynamically
    import importlib
    module = importlib.import_module(method_module)

    # Return the prepare_prompt function from that module
    return getattr(module, 'prepare_prompt')

def create_openai_batch_file(data: List[Dict], method, model: str, max_tokens: int, temperature: float, is_multimodal: bool = False) -> str:
    """Create a JSONL file for OpenAI batch processing."""
    batch_requests = []

    # Get the prepare_prompt function from the appropriate module
    prepare_prompt_func = get_prepare_prompt_function(method)

    for i, entry in enumerate(data):
        # Prepare the prompt using the specified method
        prompt_result = prepare_prompt_func(entry, is_multimodal)
        messages = prompt_result.get('messages', [])

        # Create the request with appropriate parameters for the model
        request_body = {
            "model": model,
            "messages": messages
        }

        # Use max_completion_tokens for GPT-5, o1, o3, o4 models, max_tokens for others
        if model.startswith("gpt-5") or model.startswith("o1-") or model.startswith("o3-") or model.startswith("o4-"):
            request_body["max_completion_tokens"] = max_tokens
            # Don't include temperature for reasoning models (o1, o3)
            # For GPT-5, only temperature=1 is supported, so omit it to use default
            if model.startswith("o4-"):
                request_body["temperature"] = temperature
            # GPT-5 and o1/o3 models: omit temperature to use default
        else:
            request_body["max_tokens"] = max_tokens
            request_body["temperature"] = temperature

        # Add image support for multimodal models
        if is_multimodal and entry.get('image'):
            # Handle multiple images (comma-separated)
            image_field = entry['image'].strip()
            image_paths = [img.strip() for img in image_field.split(',') if img.strip()]

            processed_images = []

            for image_path in image_paths:
                # Add datasets/MatSciBench/ prefix if not already present
                if not image_path.startswith("datasets/MatSciBench/"):
                    full_image_path = f"datasets/MatSciBench/{image_path}"
                else:
                    full_image_path = image_path

                try:
                    import base64
                    import os
                    if os.path.exists(full_image_path):
                        with open(full_image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            # Determine mime type based on file extension
                            if full_image_path.lower().endswith('.png'):
                                mime_type = "image/png"
                            elif full_image_path.lower().endswith(('.jpg', '.jpeg')):
                                mime_type = "image/jpeg"
                            else:
                                mime_type = "image/jpeg"  # Default fallback

                            image_url = f"data:{mime_type};base64,{image_data}"
                            processed_images.append({"type": "image_url", "image_url": {"url": image_url}})
                    else:
                        print(f"Warning: OpenAI batch - Image file not found: {full_image_path}")
                except Exception as e:
                    print(f"Error processing image {full_image_path}: {e}")

            # Add all images to the user message
            if processed_images:
                for message in messages:
                    if message["role"] == "user" and isinstance(message["content"], str):
                        # Convert to multimodal format with text + all images
                        content = [{"type": "text", "text": message["content"]}]
                        content.extend(processed_images)
                        message["content"] = content
                        break  # Only modify the first user message

        # Clean the QID to make it valid for custom_id (alphanumeric, underscore, hyphen only)
        clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
        # Ensure it's within 64 character limit
        if len(clean_qid) > 40:  # Leave room for prefix
            clean_qid = clean_qid[:40]

        batch_request = {
            "custom_id": f"req_{i}_{clean_qid}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body
        }

        batch_requests.append(batch_request)

    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for request in batch_requests:
        temp_file.write(json.dumps(request) + '\n')
    temp_file.close()

    return temp_file.name

def create_claude_batch_requests(data: List[Dict], method, model: str, max_tokens: int, temperature: float, is_multimodal: bool = False) -> List[Dict]:
    """Create batch requests for Claude."""
    batch_requests = []

    # Get the prepare_prompt function from the appropriate module
    prepare_prompt_func = get_prepare_prompt_function(method)

    for i, entry in enumerate(data):
        # Prepare the prompt using the specified method
        prompt_result = prepare_prompt_func(entry, is_multimodal)
        messages = prompt_result.get('messages', [])

        # Handle images for multimodal models - convert to base64 and add to messages
        if is_multimodal and entry.get('image'):
            # Handle multiple images (comma-separated)
            image_field = entry['image'].strip()
            image_paths = [img.strip() for img in image_field.split(',') if img.strip()]

            processed_images = []

            for image_path in image_paths:
                # Add datasets/MatSciBench/ prefix if not already present
                if not image_path.startswith("datasets/MatSciBench/"):
                    full_image_path = f"datasets/MatSciBench/{image_path}"
                else:
                    full_image_path = image_path

                try:
                    import base64
                    import os
                    if os.path.exists(full_image_path):
                        with open(full_image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            # Determine mime type based on file extension
                            if full_image_path.lower().endswith('.png'):
                                mime_type = "image/png"
                            elif full_image_path.lower().endswith(('.jpg', '.jpeg')):
                                mime_type = "image/jpeg"
                            else:
                                mime_type = "image/jpeg"  # Default fallback

                            processed_images.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data
                                }
                            })
                    else:
                        print(f"Warning: Claude batch - Image file not found: {full_image_path}")
                except Exception as e:
                    print(f"Error processing image {full_image_path} for Claude batch: {e}")

            # Add all images to the user message
            if processed_images:
                for message in messages:
                    if message["role"] == "user":
                        if isinstance(message["content"], str):
                            # Convert string content to multimodal format with all images
                            content = [{"type": "text", "text": message["content"]}]
                            content.extend(processed_images)
                            message["content"] = content
                        elif isinstance(message["content"], list):
                            # Add images to existing multimodal content
                            message["content"].extend(processed_images)
                        break  # Only add to the first user message

        # Convert to Claude format and extract system message
        claude_messages = []
        system_prompt = ""

        for msg in messages:
            if msg.get("role") == "system":
                # Extract system message content
                system_prompt = msg.get("content", "")
                continue  # Skip adding system messages to the messages array

            if isinstance(msg.get('content'), list):
                # Handle multimodal content
                claude_content = []
                for content_item in msg['content']:
                    if content_item['type'] == 'text':
                        claude_content.append(content_item)
                    elif content_item['type'] == 'image_url':
                        # Convert OpenAI image format to Claude format
                        image_url = content_item['image_url']['url']

                        # Handle different image URL formats
                        if image_url.startswith('data:'):
                            # Extract base64 data from data URL
                            if ';base64,' in image_url:
                                media_type_part, base64_data = image_url.split(';base64,', 1)
                                media_type = media_type_part.split(':', 1)[1]
                            else:
                                media_type = "image/jpeg"
                                base64_data = image_url

                            claude_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
                                }
                            })
                    elif content_item['type'] == 'image':
                        # Already in Claude format, keep as-is
                        claude_content.append(content_item)
                claude_messages.append({
                    "role": msg["role"],
                    "content": claude_content
                })
            else:
                claude_messages.append(msg)

        # Clean the QID to make it valid for custom_id (alphanumeric, underscore, hyphen only)
        clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
        # Ensure it's within 64 character limit
        if len(clean_qid) > 40:  # Leave room for prefix
            clean_qid = clean_qid[:40]

        # Configure request parameters
        request_params = {
            "model": model,
            "messages": claude_messages,
            "temperature": temperature
        }

        # Add system prompt if present
        if system_prompt:
            request_params["system"] = system_prompt

        # Enable extended thinking for Claude 4 models
        if "claude" in model.lower() and "4" in model:
            # Set thinking budget to max_tokens and total output to double of max_tokens
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": max_tokens
            }
            request_params["max_tokens"] = max_tokens * 2  # Double the max_tokens for total output
            request_params["temperature"] = 1  # Temperature must be 1 for extended thinking
        else:
            request_params["max_tokens"] = max_tokens

        request = {
            "custom_id": f"req_{i}_{clean_qid}",
            "params": request_params
        }

        batch_requests.append(request)

    return batch_requests

def submit_openai_batch(batch_file_path: str, description: str = "MatSciBench evaluation") -> str:
    """Submit a batch job to OpenAI and return the batch ID."""
    client = openai.OpenAI()

    # Upload the file
    with open(batch_file_path, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )

    # Create the batch job
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )

    print(f"OpenAI batch submitted with ID: {batch.id}")
    return batch.id

def submit_claude_batch(batch_requests: List[Dict], description: str = "MatSciBench evaluation") -> str:
    """Submit a batch job to Claude and return the batch ID."""
    client = anthropic.Anthropic()

    # Submit the batch using the correct API format
    batch = client.beta.messages.batches.create(
        requests=batch_requests
    )

    print(f"Claude batch submitted with ID: {batch.id}")
    return batch.id

def wait_for_openai_batch(batch_id: str, poll_interval: int = 60) -> Dict:
    """Wait for OpenAI batch to complete and return results."""
    client = openai.OpenAI()

    print(f"Waiting for OpenAI batch {batch_id} to complete...")

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch.status}")

        if batch.status == "completed":
            print("Batch completed successfully!")
            return batch
        elif batch.status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Batch failed with status: {batch.status}")

        time.sleep(poll_interval)

def wait_for_claude_batch(batch_id: str, poll_interval: int = 60) -> Dict:
    """Wait for Claude batch to complete and return results."""
    client = anthropic.Anthropic()

    print(f"Waiting for Claude batch {batch_id} to complete...")

    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        print(f"Batch processing_status: {batch.processing_status}")

        if batch.processing_status == "ended":
            print("Batch completed successfully!")
            return batch
        elif batch.processing_status in ["canceling", "canceled"]:
            raise Exception(f"Batch failed with status: {batch.processing_status}")

        time.sleep(poll_interval)

def download_openai_results(batch: Dict, data: List[Dict]) -> List[Dict]:
    """Download and parse OpenAI batch results."""
    client = openai.OpenAI()

    # Check batch completion status
    request_counts = getattr(batch, 'request_counts', None)
    if request_counts:
        print(f"Batch request counts: completed={request_counts.completed}, failed={request_counts.failed}, total={request_counts.total}")

        # If some requests failed, check error file to show what went wrong
        if request_counts.failed > 0:
            error_file_id = getattr(batch, 'error_file_id', None)
            if error_file_id:
                print(f"Some requests failed ({request_counts.failed} out of {request_counts.total}). Reading error file...")
                error_content = client.files.content(error_file_id)
                error_text = error_content.content.decode('utf-8').strip()
                error_lines = error_text.split('\n')
                for line in error_lines[:5]:  # Show first 5 errors
                    if line.strip():
                        try:
                            error_obj = json.loads(line)
                            custom_id = error_obj.get('custom_id', 'unknown')
                            if error_obj.get('response') and error_obj['response'].get('body'):
                                # Error is in response body
                                error_info = error_obj['response']['body'].get('error', {})
                                print(f"Error for {custom_id}: {error_info}")
                            elif error_obj.get('error'):
                                # Error is in error field
                                error_info = error_obj.get('error', {})
                                print(f"Error for {custom_id}: {error_info}")
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse error line: {line[:100]}... (JSON Error: {e})")

            # Only raise exception if ALL requests failed
            if request_counts.completed == 0:
                raise Exception(f"All {request_counts.failed} batch requests failed")

    # Download the results file
    result_file_id = getattr(batch, 'output_file_id', None)
    if not result_file_id:
        raise Exception("No output_file_id found in batch result")

    result = client.files.content(result_file_id)

    # Parse the results
    results = []
    result_lines = result.content.decode('utf-8').strip().split('\n')

    # Create a mapping from custom_id to original data
    id_to_data = {}
    for i, entry in enumerate(data):
        # Use the same cleaning logic as in batch creation
        clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
        if len(clean_qid) > 40:  # Leave room for prefix
            clean_qid = clean_qid[:40]
        custom_id = f"req_{i}_{clean_qid}"
        id_to_data[custom_id] = entry

    for line in result_lines:
        if line.strip():
            result_obj = json.loads(line)
            custom_id = result_obj['custom_id']

            if custom_id in id_to_data:
                original_entry = id_to_data[custom_id]

                # Check if this is a successful response or error
                if result_obj.get('response') and result_obj['response'].get('status_code') == 200:
                    response_body = result_obj['response']['body']
                    if response_body.get('choices'):
                        response_text = response_body['choices'][0]['message']['content']

                        results.append({
                            'qid': original_entry.get('qid'),
                            'question': original_entry.get('question'),
                            'correct_answer': original_entry.get('answer'),
                            'full_output': response_text,
                            'final_answer': response_text,  # This would need parsing
                            'has_image': bool(original_entry.get('image')),
                            'new_token_nums': response_body.get('usage', {}).get('completion_tokens', 0)
                        })
                elif result_obj.get('error'):
                    # Handle individual request errors
                    error_info = result_obj['error']
                    print(f"Request {custom_id} failed: {error_info}")
                else:
                    print(f"Unexpected result format for {custom_id}: {result_obj}")

    return results

def download_claude_results(batch: Dict, data: List[Dict]) -> List[Dict]:
    """Download and parse Claude batch results."""
    client = anthropic.Anthropic()

    # Get the results from the batch
    results = []

    # Create a mapping from custom_id to original data
    id_to_data = {}
    for i, entry in enumerate(data):
        # Use the same cleaning logic as in batch creation
        clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
        if len(clean_qid) > 40:  # Leave room for prefix
            clean_qid = clean_qid[:40]
        custom_id = f"req_{i}_{clean_qid}"
        id_to_data[custom_id] = entry

    # Download results using object attributes
    batch_results = {'succeeded': [], 'errored': [], 'canceled': [], 'expired': []}
    try:
        # Use the results method to get result objects
        results_iter = client.beta.messages.batches.results(batch.id)
        for result in results_iter:
            custom_id = result.custom_id
            result_type = result.result.type if hasattr(result, 'result') and result.result else None

            if result_type == 'succeeded':
                # This is a successful result
                message = result.result.message
                batch_results['succeeded'].append({
                    'custom_id': custom_id,
                    'result': {
                        'content': message.content,
                        'usage': message.usage._asdict() if hasattr(message.usage, '_asdict') else {'output_tokens': message.usage.output_tokens}
                    }
                })
            elif result_type == 'errored':
                # This is an error result
                error_info = result.result.error if hasattr(result.result, 'error') else str(result.result)
                batch_results['errored'].append({
                    'custom_id': custom_id,
                    'error': error_info
                })
                print(f"Error result for {custom_id}: {error_info}")
            elif result_type == 'canceled':
                batch_results['canceled'].append({
                    'custom_id': custom_id
                })
            elif result_type == 'expired':
                batch_results['expired'].append({
                    'custom_id': custom_id
                })
            else:
                print(f"Unknown result type: {result_type} for {custom_id}")
                print(f"Result object type: {type(result)}")
                print(f"Result attributes: {dir(result)}")
    except Exception as e:
        print(f"Error retrieving Claude batch results: {e}")
        # Fall back to trying the results_url approach
        if hasattr(batch, 'results_url') and batch.results_url:
            try:
                import requests
                response = requests.get(batch.results_url)
                batch_results = response.json()
            except Exception as url_e:
                print(f"Error retrieving results from URL: {url_e}")
                return results

    if not any(batch_results.values()):
        print("No results found in Claude batch")
        return results

    for result_type in ['succeeded', 'errored', 'canceled', 'expired']:
        if result_type in batch_results:
            for result_obj in batch_results[result_type]:
                custom_id = result_obj.get('custom_id')
                if custom_id not in id_to_data:
                    continue
                original_entry = id_to_data[custom_id]

                if result_type == 'succeeded' and result_obj.get('result'):
                    # Handle extended thinking response format for Claude 4 models
                    response_content = result_obj['result'].get('content', [])

                    # Check if this is a Claude 4 model with extended thinking based on the content structure
                    has_thinking = any(
                        (hasattr(content_block, 'type') and content_block.type == 'thinking') or
                        (isinstance(content_block, dict) and content_block.get('type') == 'thinking')
                        for content_block in response_content
                    )

                    if has_thinking:
                        thinking_content = ""
                        final_response = ""
                        for content_block in response_content:
                            # Handle both object and dict formats
                            if hasattr(content_block, 'type'):
                                # Object format
                                if content_block.type == 'thinking':
                                    thinking_content = getattr(content_block, 'content', '') or getattr(content_block, 'text', '')
                                elif content_block.type == 'text':
                                    final_response = getattr(content_block, 'content', '') or getattr(content_block, 'text', '')
                            elif isinstance(content_block, dict):
                                # Dict format
                                if content_block.get('type') == 'thinking':
                                    thinking_content = content_block.get('text', '')
                                elif content_block.get('type') == 'text':
                                    final_response = content_block.get('text', '')
                        response_text = (
                            f"<thinking>\n{thinking_content}\n</thinking>\n\n{final_response}" if thinking_content else final_response
                        )
                    else:
                        # Standard response format for non-Claude-4 models
                        if response_content and isinstance(response_content, list):
                            first_block = response_content[0]
                            if hasattr(first_block, 'text'):
                                response_text = first_block.text
                            elif isinstance(first_block, dict):
                                response_text = first_block.get('text', '')
                            else:
                                response_text = str(first_block)
                        else:
                            response_text = ""

                    results.append({
                        'qid': original_entry.get('qid'),
                        'question': original_entry.get('question'),
                        'correct_answer': original_entry.get('answer'),
                        'full_output': response_text,
                        'final_answer': response_text,  # This would need parsing
                        'has_image': bool(original_entry.get('image')),
                        'new_token_nums': result_obj['result'].get('usage', {}).get('output_tokens', 0)
                    })

    return results

def process_batch_openai(data: List[Dict], method, model: str, max_tokens: int, temperature: float, is_multimodal: bool = False) -> List[Dict]:
    """Process a batch job with OpenAI."""
    try:
        # Create batch file
        batch_file_path = create_openai_batch_file(data, method, model, max_tokens, temperature, is_multimodal)

        # Submit batch
        batch_id = submit_openai_batch(batch_file_path, f"MatSciBench evaluation - {model}")

        # Wait for completion
        completed_batch = wait_for_openai_batch(batch_id)

        # Download results
        results = download_openai_results(completed_batch, data)

        # Clean up temp file
        os.unlink(batch_file_path)

        return results

    except Exception as e:
        print(f"Batch processing failed: {e}")
        # Fall back to individual requests if batch fails
        raise e


def cancel_openai_batch(batch_id: str) -> bool:
    """Cancel an OpenAI batch job."""
    try:
        client = openai.OpenAI()
        cancelled_batch = client.batches.cancel(batch_id)
        print(f"OpenAI batch {batch_id} cancellation requested")
        print(f"Status: {cancelled_batch.status}")
        return True
    except Exception as e:
        print(f"Error cancelling OpenAI batch {batch_id}: {e}")
        return False


def cancel_claude_batch(batch_id: str) -> bool:
    """Cancel a Claude batch job."""
    try:
        client = anthropic.Anthropic()
        cancelled_batch = client.beta.messages.batches.cancel(batch_id)
        print(f"Claude batch {batch_id} cancellation requested")
        print(f"Status: {cancelled_batch.processing_status}")
        return True
    except Exception as e:
        print(f"Error cancelling Claude batch {batch_id}: {e}")
        return False

def process_batch_claude(data: List[Dict], method, model: str, max_tokens: int, temperature: float, is_multimodal: bool = False) -> List[Dict]:
    """Process a batch job with Claude."""
    try:
        # Create batch requests
        batch_requests = create_claude_batch_requests(data, method, model, max_tokens, temperature, is_multimodal)

        # Submit batch
        batch_id = submit_claude_batch(batch_requests, f"MatSciBench evaluation - {model}")

        # Wait for completion
        completed_batch = wait_for_claude_batch(batch_id)

        # Download results
        results = download_claude_results(completed_batch, data)

        return results

    except Exception as e:
        print(f"Batch processing failed: {e}")
        # Fall back to individual requests if batch fails
        raise e


def cancel_openai_batch(batch_id: str) -> bool:
    """Cancel an OpenAI batch job."""
    try:
        client = openai.OpenAI()
        cancelled_batch = client.batches.cancel(batch_id)
        print(f"OpenAI batch {batch_id} cancellation requested")
        print(f"Status: {cancelled_batch.status}")
        return True
    except Exception as e:
        print(f"Error cancelling OpenAI batch {batch_id}: {e}")
        return False


def cancel_claude_batch(batch_id: str) -> bool:
    """Cancel a Claude batch job."""
    try:
        client = anthropic.Anthropic()
        cancelled_batch = client.beta.messages.batches.cancel(batch_id)
        print(f"Claude batch {batch_id} cancellation requested")
        print(f"Status: {cancelled_batch.processing_status}")
        return True
    except Exception as e:
        print(f"Error cancelling Claude batch {batch_id}: {e}")
        return False
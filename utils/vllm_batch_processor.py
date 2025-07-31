from tqdm import tqdm
from utils.utils import extract_final_answer
from methods.prompts import SYSTEM_PROMPT, RAG_QUERY_PROMPT, RAG_SUMMARY_PROMPT, TOOL_SYSTEM_PROMPT, TOOL_FINAL_ANSWER_PROMPT, FEEDBACK_PROMPT, CORRECTION_PROMPT


def prepare_conversations_for_method(filtered_data, method_name, is_multimodal):
    """Prepare all conversations for a specific method."""
    conversations = []
    entries_with_metadata = []
    
    for entry in filtered_data:
        question_text = entry["question"]
        if entry["unit"].strip() != "":
            if entry["number_of_answers"] == "single":
                question_text += f"The unit of the answer is {entry['unit']}."
            elif entry["number_of_answers"] == "multiple":
                question_text += f"The units of each required answer are {entry['unit']}, respectively."
        
        # Parse image paths
        image_path_raw = entry.get("image", "").strip()
        image_paths = []
        if image_path_raw and image_path_raw.lower() != "nan":
            image_paths = [path.strip() for path in image_path_raw.split(',') if path.strip()]
        
        # Only pass images if the model is multimodal
        if not is_multimodal:
            image_paths = []
        
        # Build conversation based on method
        if method_name == "base":
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_text}
            ]
            conversations.append(conversation)
            entries_with_metadata.append({
                "entry": entry,
                "conversation_type": "base",
                "question_text": question_text,
                "image_paths": image_paths
            })
            
        elif method_name == "tool":
            conversation = [
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": question_text}
            ]
            conversations.append(conversation)
            entries_with_metadata.append({
                "entry": entry,
                "conversation_type": "tool_initial",
                "question_text": question_text,
                "image_paths": image_paths
            })
            
        elif method_name == "self_correction":
            # Turn 1: Initial answer generation
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_text}
            ]
            conversations.append(conversation)
            entries_with_metadata.append({
                "entry": entry,
                "conversation_type": "self_correction_initial",
                "question_text": question_text,
                "image_paths": image_paths
            })
            
        # Note: RAG and self_consistency require multiple rounds
        # For now, we'll handle them with the original sequential approach
        # They can be optimized separately with more complex batching logic
        
    return conversations, entries_with_metadata


def process_batch_vllm(filtered_data, method, model_name, max_tokens, temperature, llm, sampling_params, is_multimodal):
    """Process all data entries using vLLM batch processing."""
    method_name = method.__name__
    responses = []
    
    if method_name in ["base", "tool_augmentation", "self_correction"]:
        # For methods that can be easily batched
        conversations, entries_with_metadata = prepare_conversations_for_method(
            filtered_data, method_name.replace("_augmentation", ""), is_multimodal
        )
        
        if conversations:
            print(f"Processing {len(conversations)} conversations in batch...")
            
            # Process all conversations in a single vLLM batch for the first turn
            outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
            
            # Handle multi-turn methods that need batch processing for subsequent turns
            if method_name == "tool_augmentation":
                responses = handle_tool_augmentation_batch_all(
                    outputs, entries_with_metadata, llm, sampling_params
                )
            elif method_name == "self_correction":
                responses = handle_self_correction_batch_all(
                    outputs, entries_with_metadata, llm, sampling_params
                )
            else:
                # Process results for single-turn methods
                for i, (output, metadata) in enumerate(zip(outputs, entries_with_metadata)):
                    try:
                        entry = metadata["entry"]
                        question_text = metadata["question_text"]
                        image_paths = metadata["image_paths"]
                        
                        full_output = output.outputs[0].text.strip()
                        new_token_nums = len(output.outputs[0].token_ids)
                        
                        final_answer = extract_final_answer(full_output) if full_output else ""
                        
                        result = {
                            "qid": entry.get("qid", ""),
                            "question_type": entry["type"],
                            "question": question_text,
                            "full_output": full_output,
                            "final_answer": final_answer,
                            "correct_solution": entry.get("solution", ""),
                            "correct_answer": str(entry["answer"]).strip() if entry["answer"] is not None else "",
                            "unit": entry.get("unit", ""),
                            "number_of_answers": entry.get("number_of_answers", ""),
                            "domain": entry.get("domain", ""),
                            "new_token_nums": new_token_nums,
                            "image": entry.get("image", "").strip(),
                            "error": None,
                            "has_image": bool(image_paths)
                        }
                        responses.append(result)
                    
                    except Exception as e:
                        print(f"Error processing entry {entry.get('qid', 'unknown')}: {e}")
                        # Add error response
                        error_result = {
                            "qid": entry.get("qid", ""),
                            "question_type": entry.get("type", ""),
                            "question": question_text if 'question_text' in locals() else "",
                            "full_output": "",
                            "final_answer": "",
                            "correct_solution": "",
                            "correct_answer": "",
                            "unit": "",
                            "number_of_answers": "",
                            "domain": "",
                            "new_token_nums": 0,
                            "image": "",
                            "error": str(e),
                            "has_image": False
                        }
                        responses.append(error_result)
    
    else:
        # For complex methods (RAG, self_consistency), fall back to sequential processing
        print(f"Method {method_name} requires sequential processing due to multiple rounds...")
        for entry in tqdm(filtered_data, desc="Processing entries"):
            try:
                result = method(
                    entry, model_name, max_tokens, temperature, "vllm", 
                    llm, sampling_params, is_multimodal
                )
                if result:
                    result["has_image"] = bool((result.get("image_path") and result["image_path"].strip()) or 
                                            (result.get("image") and result["image"].strip()))
                    responses.append(result)
            except Exception as e:
                print(f"Error processing entry {entry.get('qid', 'unknown')}: {e}")
    
    return responses


def handle_tool_augmentation_batch_all(initial_outputs, entries_with_metadata, llm, sampling_params):
    """Handle tool augmentation follow-up processing with batching for all entries."""
    import re
    from utils.python_executor import PythonExecutor
    
    responses = []
    
    # Extract initial outputs and execute code for all entries
    processed_results = []
    for output, metadata in zip(initial_outputs, entries_with_metadata):
        initial_output = output.outputs[0].text.strip()
        initial_tokens = len(output.outputs[0].token_ids)
        question_text = metadata["question_text"]
        
        # Extract and execute Python code from the initial response
        python_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(python_pattern, initial_output, re.DOTALL)
        
        code_executed = None
        if matches:
            # Execute the Python code
            executor = PythonExecutor()
            code_to_execute = matches[-1]  # Take the last code block
            
            try:
                execution_result = executor.execute(code_to_execute)
                if execution_result["success"]:
                    code_executed = f"Code executed successfully:\n```python\n{code_to_execute}\n```\n\nOutput:\n{execution_result['output']}"
                else:
                    code_executed = f"Code execution failed:\n```python\n{code_to_execute}\n```\n\nError:\n{execution_result['error']}"
            except Exception as e:
                code_executed = f"Code execution failed:\n```python\n{code_to_execute}\n```\n\nError:\n{str(e)}"
        
        processed_results.append({
            'initial_output': initial_output,
            'initial_tokens': initial_tokens,
            'code_executed': code_executed,
            'metadata': metadata
        })
    
    # Prepare follow-up conversations for entries that have executed code
    followup_conversations = []
    entries_with_followup = []
    
    print("Processing tool augmentation follow-up batch...")
    for result in processed_results:
        if result['code_executed'] is not None:
            metadata = result['metadata']
            question_text = metadata["question_text"]
            initial_output = result['initial_output']
            code_executed = result['code_executed']
            
            followup_conversation = [
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": initial_output},
                {"role": "user", "content": f"{code_executed}\n\n{TOOL_FINAL_ANSWER_PROMPT}"}
            ]
            followup_conversations.append(followup_conversation)
            entries_with_followup.append(result)
    
    # Process all follow-up conversations in batch
    if followup_conversations:
        followup_outputs = llm.chat(followup_conversations, sampling_params=sampling_params, use_tqdm=True)
    else:
        followup_outputs = []
    
    # Combine results
    followup_idx = 0
    for result in processed_results:
        try:
            metadata = result['metadata']
            entry = metadata["entry"]
            question_text = metadata["question_text"]
            image_paths = metadata["image_paths"]
            
            initial_output = result['initial_output']
            initial_tokens = result['initial_tokens']
            
            if result['code_executed'] is not None:
                # Has follow-up response
                followup_response = followup_outputs[followup_idx].outputs[0].text.strip()
                followup_tokens = len(followup_outputs[followup_idx].outputs[0].token_ids)
                followup_idx += 1
                
                # Combine outputs
                full_output = initial_output + "\n\n" + "-" * 60 + "\n\n" + result['code_executed'] + "\n\n" + followup_response
                total_tokens = initial_tokens + followup_tokens
            else:
                # No code to execute, use initial output
                full_output = initial_output
                total_tokens = initial_tokens
            
            final_answer = extract_final_answer(full_output) if full_output else ""
            
            result = {
                "qid": entry.get("qid", ""),
                "question_type": entry["type"],
                "question": question_text,
                "full_output": full_output,
                "final_answer": final_answer,
                "correct_solution": entry.get("solution", ""),
                "correct_answer": str(entry["answer"]).strip() if entry["answer"] is not None else "",
                "unit": entry.get("unit", ""),
                "number_of_answers": entry.get("number_of_answers", ""),
                "domain": entry.get("domain", ""),
                "new_token_nums": total_tokens,
                "image": entry.get("image", "").strip(),
                "error": None,
                "has_image": bool(image_paths)
            }
            responses.append(result)
            
        except Exception as e:
            print(f"Error processing entry {metadata['entry'].get('qid', 'unknown')}: {e}")
            # Add error response
            error_result = {
                "qid": metadata['entry'].get("qid", ""),
                "question_type": metadata['entry'].get("type", ""),
                "question": metadata["question_text"],
                "full_output": "",
                "final_answer": "",
                "correct_solution": "",
                "correct_answer": "",
                "unit": "",
                "number_of_answers": "",
                "domain": "",
                "new_token_nums": 0,
                "image": "",
                "error": str(e),
                "has_image": False
            }
            responses.append(error_result)
    
    return responses


def handle_self_correction_batch_all(initial_outputs, entries_with_metadata, llm, sampling_params):
    """Handle self-correction multi-turn processing with batching for all entries."""
    from methods.prompts import FEEDBACK_PROMPT, CORRECTION_PROMPT
    
    responses = []
    
    # Extract initial outputs and prepare for next turns
    initial_results = []
    for output, metadata in zip(initial_outputs, entries_with_metadata):
        initial_output = output.outputs[0].text.strip()
        initial_tokens = len(output.outputs[0].token_ids)
        initial_results.append({
            'initial_output': initial_output,
            'initial_tokens': initial_tokens,
            'metadata': metadata
        })
    
    # Step 2: Prepare all review conversations and process in batch
    print("Processing review batch...")
    review_conversations = []
    for result in initial_results:
        metadata = result['metadata']
        question_text = metadata["question_text"]
        initial_output = result['initial_output']
        
        review_conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": FEEDBACK_PROMPT}
        ]
        review_conversations.append(review_conversation)
    
    # Process all review conversations in batch
    review_outputs = llm.chat(review_conversations, sampling_params=sampling_params, use_tqdm=True)
    
    # Step 3: Prepare all correction conversations and process in batch
    print("Processing correction batch...")
    correction_conversations = []
    for i, result in enumerate(initial_results):
        metadata = result['metadata']
        question_text = metadata["question_text"]
        initial_output = result['initial_output']
        review_output = review_outputs[i].outputs[0].text.strip()
        
        correction_conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": FEEDBACK_PROMPT},
            {"role": "assistant", "content": review_output},
            {"role": "user", "content": CORRECTION_PROMPT}
        ]
        correction_conversations.append(correction_conversation)
    
    # Process all correction conversations in batch
    correction_outputs = llm.chat(correction_conversations, sampling_params=sampling_params, use_tqdm=True)
    
    # Combine all results
    for i, result in enumerate(initial_results):
        try:
            metadata = result['metadata']
            entry = metadata["entry"]
            question_text = metadata["question_text"]
            image_paths = metadata["image_paths"]
            
            initial_output = result['initial_output']
            initial_tokens = result['initial_tokens']
            review_output = review_outputs[i].outputs[0].text.strip()
            review_tokens = len(review_outputs[i].outputs[0].token_ids)
            final_output = correction_outputs[i].outputs[0].text.strip()
            final_tokens = len(correction_outputs[i].outputs[0].token_ids)
            
            # Combine all outputs to match the original self_correction format
            full_output = f"{initial_output}\n\n{FEEDBACK_PROMPT}\n\n{review_output}\n\n{CORRECTION_PROMPT}\n\n{final_output}"
            total_tokens = initial_tokens + review_tokens + final_tokens
            
            # Extract answers
            initial_answer = extract_final_answer(initial_output) if initial_output else ""
            final_answer = extract_final_answer(final_output) if final_output else ""
            
            result = {
                "qid": entry.get("qid", ""),
                "question_type": entry["type"],
                "question": question_text,
                "full_output": full_output,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "correct_solution": entry.get("solution", ""),
                "correct_answer": str(entry["answer"]).strip() if entry["answer"] is not None else "",
                "unit": entry.get("unit", ""),
                "number_of_answers": entry.get("number_of_answers", ""),
                "domain": entry.get("domain", ""),
                "new_token_nums": total_tokens,
                "image": entry.get("image", "").strip(),
                "error": None,
                "has_image": bool(image_paths)
            }
            responses.append(result)
            
        except Exception as e:
            print(f"Error processing entry {metadata['entry'].get('qid', 'unknown')}: {e}")
            # Add error response
            error_result = {
                "qid": metadata['entry'].get("qid", ""),
                "question_type": metadata['entry'].get("type", ""),
                "question": metadata["question_text"],
                "full_output": "",
                "initial_answer": "",
                "final_answer": "",
                "correct_solution": "",
                "correct_answer": "",
                "unit": "",
                "number_of_answers": "",
                "domain": "",
                "new_token_nums": 0,
                "image": "",
                "error": str(e),
                "has_image": False
            }
            responses.append(error_result)
    
    return responses
from utils import generate_with_api, extract_final_answer
from methods.prompts import SYSTEM_PROMPT, FEEDBACK_PROMPT, CORRECTION_PROMPT

def self_correction(entry, model, max_tokens, temperature, model_type, llm=None, sampling_params=None, is_multimodal=False):
    """Process a single entry with self-correction for any model type"""
    try:
        question_text = entry["question"]
        if entry["unit"].strip() != "":
            if entry["number_of_answers"] == "single":
                question_text += f"The unit of the answer is {entry['unit']}."
            elif entry["number_of_answers"] == "multiple":
                question_text += f"The units of each required answer are {entry['unit']}, respectively."
            else:
                raise ValueError(f"Invalid number of answers: {entry['number_of_answers']}")
            
        q_type = entry["type"]
        correct = str(entry["answer"]).strip() if entry["answer"] is not None else ""
        domain = entry.get("domain", "")
        correct_solution = entry.get("solution", "")
        image_path_raw = entry.get("image", "").strip()
        number_of_answers = entry.get("number_of_answers", "")
        unit = entry.get("unit", "")
        # Parse multiple image paths if present (comma-separated)
        image_paths = []
        if image_path_raw and image_path_raw.lower() != "nan":
            # Split by comma and strip whitespace from each path
            image_paths = [path.strip() for path in image_path_raw.split(',') if path.strip()]
        
        # Only pass images if the model is multimodal
        if not is_multimodal:
            image_paths = []
        
        # Step 1: Generate initial answer
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        if model_type == "vllm":  # vLLM model
            output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
            initial_output = output[0].outputs[0].text.strip()
            initial_token_nums = len(output[0].outputs[0].token_ids)
        else:
            initial_output, initial_token_nums = generate_with_api(
                model_type,
                model,
                conversation,
                max_tokens,
                temperature,
                image_paths
            )
        
        # Step 2: Ask for review and identification of problems
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": FEEDBACK_PROMPT}
        ]

        if model_type == "vllm":
            output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
            review_output = output[0].outputs[0].text.strip()
            review_token_nums = len(output[0].outputs[0].token_ids)
        else:
            review_output, review_token_nums = generate_with_api(
                model_type,
                model,
                conversation,
                max_tokens,
                temperature,
                image_paths
            )
        
        # Step 3: Ask for improved answer based on identified problems
                
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": FEEDBACK_PROMPT},
            {"role": "assistant", "content": review_output},
            {"role": "user", "content": CORRECTION_PROMPT}
        ]

        if model_type == "vllm":
            output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
            final_output = output[0].outputs[0].text.strip()
            final_token_nums = len(output[0].outputs[0].token_ids)
        else:
            final_output, final_token_nums = generate_with_api(
                model_type,
                model,
                conversation,
                max_tokens,
                temperature,
                image_paths
            )
        
        # Concatenate all outputs for the full conversation
        full_output = f"{initial_output}\n\n{FEEDBACK_PROMPT}\n\n{review_output}\n\n{CORRECTION_PROMPT}\n\n{final_output}"
        new_token_nums = initial_token_nums + review_token_nums + final_token_nums
        
        initial_answer = extract_final_answer(initial_output)
        # Extract the final answer from the improved response
        final_answer = extract_final_answer(final_output)

        return {
            "qid": entry.get("qid", ""),
            "question_type": q_type,
            "question": question_text,
            "full_output": full_output,
            "initial_answer": initial_answer,
            "final_answer": final_answer,
            "correct_solution": correct_solution,
            "correct_answer": correct,  
            "unit": unit,
            "number_of_answers": number_of_answers,
            "domain": domain,
            "new_token_nums": new_token_nums,
            "image": image_path_raw
        }
    
    except Exception as err:
        print(f"Error processing question {entry.get('qid', 'unknown')}: {err}")
        return None
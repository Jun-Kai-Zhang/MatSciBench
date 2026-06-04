from utils import generate_with_api, extract_final_answer
from utils.image_inputs import entry_images, image_count, image_summary
from methods.prompts import SYSTEM_PROMPT, FEEDBACK_PROMPT, CORRECTION_PROMPT

def prepare_prompt(entry, is_multimodal=False):
    """Prepare the prompt for batch processing"""
    question_text = entry["question"]
    if entry["unit"].strip() != "":
        if entry["number_of_answers"] == "single":
            question_text += f"The unit of the answer is {entry['unit']}."
        elif entry["number_of_answers"] == "multiple":
            question_text += f"The units of each required answer are {entry['unit']}, respectively."
        else:
            raise ValueError(f"Invalid number of answers: {entry['number_of_answers']}")

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text}
    ]

    return {"messages": conversation}

def self_correction(entry, model, max_tokens, temperature, is_multimodal=False):
    """Process a single entry with self-correction."""
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
        images = entry_images(entry)
        number_of_answers = entry.get("number_of_answers", "")
        unit = entry.get("unit", "")
        
        # Only pass images if the model is multimodal
        if not is_multimodal:
            images = []
        
        # Step 1: Generate initial answer
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        initial_output, initial_token_nums = generate_with_api(
            model,
            conversation,
            max_tokens,
            temperature,
            images
        )
        
        # Step 2: Ask for review and identification of problems
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": FEEDBACK_PROMPT}
        ]

        review_output, review_token_nums = generate_with_api(
            model,
            conversation,
            max_tokens,
            temperature,
            images
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

        final_output, final_token_nums = generate_with_api(
            model,
            conversation,
            max_tokens,
            temperature,
            images
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
            "image": image_summary(entry),
            "image_count": image_count(entry),
        }
    
    except Exception as err:
        print(f"Error processing question {entry.get('qid', 'unknown')}: {err}")
        return None

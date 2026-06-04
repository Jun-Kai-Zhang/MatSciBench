from utils import generate_with_api, extract_final_answer
from utils.image_inputs import entry_images, image_count, image_summary

from methods.prompts import SYSTEM_PROMPT

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



def base(entry, model, max_tokens, temperature, is_multimodal=False):
    """Process a single entry through the configured OpenAI-compatible API."""
    try:
        question_text = entry["question"]
        if entry["unit"].strip() != "":
            if entry["number_of_answers"] == "single":
                question_text += f"The unit of the answer is {entry['unit']}."
            elif entry["number_of_answers"] == "multiple":
                question_text += f"The units of each required answer are {entry['unit']}, respectively."
            else:
                raise ValueError(f"Invalid number of answers: {entry['number_of_answers']}")
            
        correct = str(entry["answer"]).strip() if entry["answer"] is not None else ""
        domain = entry.get("domain", "")
        correct_solution = entry.get("solution", "")
        images = entry_images(entry)
        number_of_answers = entry.get("number_of_answers", "")
        unit = entry.get("unit", "")
        
        # Only pass images if the model is multimodal
        if not is_multimodal:
            images = []
            
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        full_output, new_token_nums = generate_with_api(
            model,
            conversation,
            max_tokens,
            temperature,
            images
        )

        final_answer = extract_final_answer(full_output) if full_output else ""

        return {
            "qid": entry.get("qid", ""),
            "question_type": entry["type"],
            "question": question_text,
            "full_output": full_output,
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
        return {
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
            "error": str(err)  # Include the error message from the exception
        }

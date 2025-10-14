from utils import generate_with_api, extract_final_answer
from utils.vllm_api import generate_with_vllm

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



def base(entry, model, max_tokens, temperature, model_type, llm=None, sampling_params=None, is_multimodal=False):
    """Process a single entry for any model type in parallel"""
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
            
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]
        # print("image path: ", image_paths)
        if model_type == "vllm":  # vLLM model
            response = generate_with_vllm(llm, sampling_params, conversation, image_paths)
            full_output = response["text"].strip()
            new_token_nums = response["token_ids"]
            error = None
        else:
            full_output, new_token_nums = generate_with_api(
                model_type,
                model,
                conversation,
                max_tokens,
                temperature,
                image_paths
            )
            # Extract error message if present (for Gemini models)
            error = None
            if model_type == "gemini" and full_output.startswith("Error:"):
                error_lines = full_output.split("\n")
                error = error_lines[0].replace("Error:", "").strip()
                full_output = "\n".join(error_lines[1:]).strip() if len(error_lines) > 1 else ""

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
            "image": image_path_raw,
            "error": error  # Include error message in the return value
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



# def cot(entry, model, max_tokens, temperature, model_type, llm=None, sampling_params=None, is_multimodal=False):
#     """Process a single entry for any model type in parallel"""
#     try:
#         question_text = entry["question"]
#         q_type = entry["type"]
#         correct = str(entry["answer"]).strip() if entry["answer"] is not None else ""
#         domain = entry.get("domain", "")
#         correct_solution = entry.get("solution", "")
#         image_path_raw = entry.get("image", "").strip()
#         number_of_answers = entry.get("number_of_answers", "")
#         unit = entry.get("unit", "")
#         # Parse multiple image paths if present (comma-separated)
#         image_paths = []
#         if image_path_raw and image_path_raw.lower() != "nan":
#             # Split by comma and strip whitespace from each path
#             image_paths = [path.strip() for path in image_path_raw.split(',') if path.strip()]
        
#         # Only pass images if the model is multimodal
#         if not is_multimodal:
#             image_paths = []
            
#         conversation = [
#             {"role": "system", "content": COT_SYSTEM_PROMPT},
#             {"role": "user", "content": question_text}
#         ]

#         if model_type == "vllm":  # vLLM model
#             output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
#             full_output = output[0].outputs[0].text.strip()
#             new_token_nums = len(output[0].outputs[0].token_ids)
#         else:
#             full_output, new_token_nums = generate_with_api(
#                 model_type,
#                 model,
#                 conversation,
#                 max_tokens,
#                 temperature,
#                 image_paths
#             )

#         final_answer = extract_final_answer(full_output)

#         return {
#             "qid": entry.get("qid", ""),
#             "question_type": q_type,
#             "question": question_text,
#             "full_output": full_output,
#             "final_answer": final_answer,
#             "correct_solution": correct_solution,
#             "correct_answer": correct,
#             "unit": unit,
#             "number_of_answers": number_of_answers,
#             "domain": domain,
#             "new_token_nums": new_token_nums
#         }
    
#     except Exception as err:
#         print(f"Error processing question {entry.get('qid', 'unknown')}: {err}")
#         return None



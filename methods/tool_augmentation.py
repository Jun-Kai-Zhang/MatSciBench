import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import threading
import traceback  # Added for detailed error reporting

from utils import generate_with_api, extract_final_answer
from utils.image_inputs import entry_images, image_count, image_summary
from methods.prompts import TOOL_SYSTEM_PROMPT, TOOL_FINAL_ANSWER_PROMPT
from utils.python_executor import PythonExecutor

# Add a threading lock for synchronized printing
print_lock = threading.Lock()

def prepare_prompt(entry, is_multimodal=False):
    """Prepare the prompt for batch processing (Round 1 only)"""
    question_text = entry["question"]
    if entry["unit"].strip() != "":
        if entry["number_of_answers"] == "single":
            question_text += f"The unit of the answer is {entry['unit']}."
        elif entry["number_of_answers"] == "multiple":
            question_text += f"The units of each required answer are {entry['unit']}, respectively."
        else:
            raise ValueError(f"Invalid number of answers: {entry['number_of_answers']}")

    conversation = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question_text}
    ]

    return {"messages": conversation}

def tool_augmentation(entry, model, max_tokens, temperature, is_multimodal=False):
    """Tool augmentation is a method that wrte python code to augment the model's ability to solve problems."""
    try:
        qid = entry.get('qid', 'unknown')
        # with print_lock:
        #     print(f"\n>> Starting Tool Augmentation for question {qid}", flush=True)
        
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
            
        conversation = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        # with print_lock:
        #     print(f">> Generating response for question {qid}...", flush=True)

        full_output, new_token_nums = generate_with_api(
            model,
            conversation,
            max_tokens,
            temperature,
            images
        )
            
        # with print_lock:
        #     print(f">> Executing code for question {qid}...", flush=True)
            
        # First extract code, then only execute if code was found
        code_blocks = extract_code_blocks(full_output)
        
        if not code_blocks:
            # No code found, extract answer directly from model output
            final_answer = extract_final_answer(full_output)
            code_executed = ""
        else:
            # Execute all code blocks and combine their results
            code_executed = ""
            for i, code_block in enumerate(code_blocks):
                block_result = execute_code(code_block)
                code_executed += f"Code block {i+1} execution result:\n{block_result}\n\n"
            
            # Send execution results back to the LLM for final answer
            followup_conversation = conversation.copy()
            followup_conversation.append({"role": "assistant", "content": full_output})
            followup_conversation.append({"role": "user", "content": TOOL_FINAL_ANSWER_PROMPT.format(code_executed=code_executed)})
            
            full_output += "\n\n" + "-" * 60
            full_output += "\n\n" + code_executed
            
            followup_response, additional_tokens = generate_with_api(
                model,
                followup_conversation,
                max_tokens,
                temperature,
                images
            )
            new_token_nums += additional_tokens
                
            final_answer = extract_final_answer(followup_response)
            full_output += "\n\n" + "-" * 60
            full_output += f"\n\n{followup_response}"
        # Use thread-safe synchronized output with explicit flushing
        # with print_lock:
        #     print(f"\n{'=' * 60}")
        #     print(f"QUESTION ID: {qid}")
        #     print(f"QUESTION: {question_text[:100]}...")
        #     print(f"MODEL OUTPUT: {full_output}")
        #     print(f"CODE EXECUTION RESULT: {code_executed[:200]}...")
        #     print(f"FINAL ANSWER: {final_answer}")
        #     print(f"{'=' * 60}\n", flush=True)
        
        return {
            "qid": qid,
            "question_type": q_type,
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
        with print_lock:
            print(f"ERROR processing question {entry.get('qid', 'unknown')}: {err}", flush=True)
            print(f"TRACEBACK: {traceback.format_exc()}", flush=True)
        return None

def extract_code_blocks(full_output):
    """Extract Python code blocks from the model output."""
    return re.findall(r"```python(.*?)```", full_output, re.DOTALL)

def execute_code(code_str, timeout=30):
    """Execute the extracted code using PythonExecutor and return the output."""
    executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=timeout)
    result, report = executor.apply(code_str)
    return result

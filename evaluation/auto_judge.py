import concurrent.futures
from tqdm import tqdm
import google.generativeai as genai
import os
import re
import signal
import time
import psutil
import multiprocessing
from evaluation.prompts import judge_user_prompt, JUDGE_SYSTEM_PROMPT
from utils import extract_final_answer, extract_number, generate_with_api
from evaluation.grader import math_equal


def judge_response_with_gemini(model_answer, correct_answer, question, api_key):
    """Use Gemini to judge if a response is correct"""
    # If the model answer is empty, return incorrect directly
    if not model_answer or model_answer.strip() == "":
        return {"is_correct": False, "judge_reasoning": "Model answer is empty."}
    
    try:
        # Configure API key locally for this function to be thread-safe
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", None)
        if not api_key:
            print("Warning: No Gemini API key available for judging questions.")
            return
        genai.configure(api_key=api_key)
        model = "gemini-2.0-flash"
        
        conversation = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_user_prompt(correct_answer, model_answer, question)}
        ]
        
        response, _ = generate_with_api("gemini", model, conversation, 4096, 0.0, [])

        # Extract the decision from the LaTeX box
        decision = extract_final_answer(response).lower()
        # Check if the decision is "correct"
        # Clean up the decision by removing extra spaces and normalizing
        decision = decision.strip().replace('"', '')
        # Check if the decision is "correct"
        return {"is_correct": decision == "correct", "judge_reasoning": response}
    
    except Exception as e:
        print(f"Error using Gemini to judge question: {e}")
        return {"is_correct": False, "judge_reasoning": f"Error: {str(e)}"}


def _rule_judge_worker(model_answer, correct_answer, multiple):
    """Worker function to handle rule-based judgment with extreme timeout protection"""
    model_str = model_answer.strip() if isinstance(model_answer, str) else str(model_answer)
    correct_str = correct_answer.strip() if isinstance(correct_answer, str) else str(correct_answer)
    
    if multiple or model_str.startswith('(') or correct_str.startswith('('):
        # Extract individual answers from the multiple answer format
        try:
            # Strip the first and last parentheses, then split by comma
            model_str = model_answer.strip()
            correct_str = correct_answer.strip()
            
            # Remove outer parentheses if they exist
            if model_str.startswith('(') and model_str.endswith(')'):
                model_str = model_str[1:-1]
            if correct_str.startswith('(') and correct_str.endswith(')'):
                correct_str = correct_str[1:-1]
            
            # Split by comma and strip spaces
            model_answers = [ans.strip() for ans in model_str.split(',')]
            correct_answers = [ans.strip() for ans in correct_str.split(',')]
            
            # Check if we have the same number of answers
            if len(model_answers) != len(correct_answers):
                return {"is_correct": False, "judge_reasoning": f"Number of answers doesn't match. Model provided {len(model_answers)} answers, expected {len(correct_answers)}."}
            
            # Normalize scientific notation in all answers
            model_answers = [normalize_scientific_notation(ans) for ans in model_answers]
            correct_answers = [normalize_scientific_notation(ans) for ans in correct_answers]
            
            # Check each answer
            all_correct = True
            incorrect_answers = []
            for i, (m_ans, c_ans) in enumerate(zip(model_answers, correct_answers)):
                if not math_equal(m_ans, c_ans):
                    all_correct = False
                    incorrect_answers.append(f"Answer #{i+1}: {m_ans} ≠ {c_ans}")
            
            if all_correct:
                return {"is_correct": True, "judge_reasoning": f"All answers match: {model_answers} = {correct_answers}"}
            else:
                return {"is_correct": False, "judge_reasoning": f"Some answers don't match: {', '.join(incorrect_answers)}"}
        
        except Exception as e:
            return {"is_correct": False, "judge_reasoning": f"Error parsing multiple answers: {str(e)}. Model answer: {model_answer}, Correct answer: {correct_answer}"}
    else:
        # If not multiple, just use math_equal directly
        model_answer = normalize_scientific_notation(model_answer)
        correct_answer = normalize_scientific_notation(correct_answer)
        is_correct = math_equal(model_answer, correct_answer)
        if is_correct:
            return {"is_correct": True, "judge_reasoning": f"Answer: {model_answer} = {correct_answer}"}
        else:
            return {"is_correct": False, "judge_reasoning": f"Answer: {model_answer} ≠ {correct_answer}"}


def _rule_judge_process_worker(model_answer, correct_answer, multiple, return_queue):
    """Wrapper that runs _rule_judge_worker in a separate process and sends the
    result back through a multiprocessing Queue so the parent can enforce a hard
    timeout by killing the process if needed."""
    try:
        result = _rule_judge_worker(model_answer, correct_answer, multiple)
    except Exception as e:
        result = {
            "is_correct": False,
            "judge_reasoning": f"Error in worker: {str(e)}"
        }
    return_queue.put(result)


def judge_response_with_rule(model_answer, correct_answer, multiple=False, timeout=20):
    """Judge if a response is correct using the rule‑based method.

    The evaluation is executed in a *separate process* so we can forcibly
    terminate it if it exceeds the specified timeout.  This guarantees that
    misbehaving or long‑running checks cannot hang the main program.
    """
    # Fast path for empty answers
    if not model_answer or model_answer.strip() == "":
        return {"is_correct": False, "judge_reasoning": "Model answer is empty."}

    # --- Run the worker in its own process ---------------------------------
    ctx = multiprocessing.get_context("spawn")      # safer than fork with 3rd‑party libs
    return_queue = ctx.Queue()
    proc = ctx.Process(
        target=_rule_judge_process_worker,
        args=(model_answer, correct_answer, multiple, return_queue),
    )
    proc.start()
    proc.join(timeout)

    # --- Handle timeout -----------------------------------------------------
    if proc.is_alive():
        proc.terminate()        # kill the worker
        proc.join()
        return {
            "is_correct": False,
            "judge_reasoning": (
                f"Rule-based judgment timed out after {timeout} seconds. "
                f"Model answer: {model_answer}, Correct answer: {correct_answer}"
            )
        }

    # --- Retrieve result ----------------------------------------------------
    if not return_queue.empty():
        return return_queue.get()

    # Fallback: process ended without sending a result
    return {
        "is_correct": False,
        "judge_reasoning": (
            "Worker process finished but did not return a result. "
            f"Model answer: {model_answer}, Correct answer: {correct_answer}"
        )
    }


def normalize_scientific_notation(answer_str):
    """
    Normalizes scientific notation in answers to ensure consistent comparison.
    Handles both programming formats (3.0e-9, 3.0E-9) and LaTeX formats (5.582 \times 10^{23}).
    """
    if not isinstance(answer_str, str):
        return answer_str
        
    # Trim whitespace and handle empty strings
    answer_str = answer_str.strip()
    if not answer_str:
        return answer_str
    
    try:
        # Case 1: Standard scientific notation (e.g., 3.0e-9)
        if re.search(r'^[+-]?\d+\.?\d*[eE][+-]?\d+$', answer_str):
            return str(float(answer_str))
            
        # Case 2: Super flexible scientific notation matcher
        # Just look for a number followed by 10^ and another number
        scientific_pattern = r'(\d+\.?\d*)(?:.*?)10\^{?([+-]?\d+)}?'
        match = re.search(scientific_pattern, answer_str)
        if match:
            coefficient = float(match.group(1))
            exponent = int(match.group(2))
            # Convert to a float and then to string
            return str(coefficient * (10 ** exponent))
            
        # If no patterns match, return the original string
        return answer_str
    except (ValueError, TypeError):
        # If parsing fails, return the original string
        return answer_str


def judge_single_response(pred, gemini_api_key=None, use_llm=True, rule_timeout=20):
    """Judge if a response is correct"""
    # print(pred["final_answer"], pred["correct_answer"])
    judge_result_rule = judge_response_with_rule(pred["final_answer"], pred["correct_answer"], 
                                               pred.get("multiple", False), timeout=rule_timeout)
    # print(judge_result_rule)
    # Always store the rule-based judgment in separate fields
    pred["rule_is_correct"] = judge_result_rule["is_correct"]
    pred["rule_judge_reasoning"] = judge_result_rule["judge_reasoning"]
    
    # Check if this is a formula question - only use LLM for formula questions
    is_formula_question = pred.get("question_type", "").upper() == "FORMULA"
    
    # Use LLM only if requested AND it's a formula question
    if use_llm and is_formula_question:
        judge_result_llm = judge_response_with_gemini(pred["final_answer"], pred["correct_answer"], pred["question"], gemini_api_key)
        # Store the LLM judgment in separate fields
        pred["llm_is_correct"] = judge_result_llm["is_correct"]
        pred["llm_judge_reasoning"] = judge_result_llm["judge_reasoning"]
        
        # If LLM judge returns an empty response, use rule-based correctness
        if not judge_result_llm.get("judge_reasoning") or judge_result_llm.get("judge_reasoning", "").strip() == "":
            pred["is_correct"] = judge_result_rule["is_correct"]
            pred["judge_reasoning"] = f"LLM response was empty. Using rule-based judgment: {judge_result_rule['judge_reasoning']}"
        else:
            # Use LLM judgment for formula questions when they disagree
            pred["is_correct"] = judge_result_llm["is_correct"]
            pred["judge_reasoning"] = f"LLM: {judge_result_llm['judge_reasoning']}\nRule: {judge_result_rule['judge_reasoning']}"
    else:
        # Only use rule-based judgment when use_llm is False or it's not a formula question
        pred["is_correct"] = judge_result_rule["is_correct"]
        if is_formula_question:
            pred["judge_reasoning"] = f"Rule (formula): {judge_result_rule['judge_reasoning']}"
        else:
            pred["judge_reasoning"] = f"Rule: {judge_result_rule['judge_reasoning']}"

    # Preserve the error field if it exists
    if "error" in pred:
        pred["error"] = pred["error"]
    else:
        pred["error"] = None

    return pred

def judge_responses(predictions, gemini_api_key=None, max_workers=8, use_llm=True, rule_timeout=240):
    """Process all judgments in parallel"""
    
    # First, process all final answers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit judging tasks for final answers
        future_to_index = {}
        for i, pred in enumerate(predictions):
            # If the model answer is empty, mark as incorrect directly without API call
            if not pred.get("final_answer") or pred.get("final_answer", "").strip() == "":
                predictions[i]["is_correct"] = False
                predictions[i]["judge_reasoning"] = "Model answer is empty."
                # Preserve error field if it exists
                if "error" in pred:
                    predictions[i]["error"] = pred["error"]
                continue
                
            future = executor.submit(
                judge_single_response,
                pred,
                gemini_api_key,
                use_llm,
                rule_timeout
            )
            future_to_index[future] = i
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Judging final answers"):
            index = future_to_index[future]
            try:
                updated_pred = future.result()
                predictions[index] = updated_pred
            except Exception as e:
                print(f"Error judging question {predictions[index].get('qid', 'unknown')}: {e}")
                predictions[index]["is_correct"] = False
                predictions[index]["judge_reasoning"] = f"Error during judging: {str(e)}"
                # Preserve error field if it exists
                if "error" in predictions[index]:
                    predictions[index]["error"] = predictions[index]["error"]
    
    # Now, process all initial answers where available
    initial_answers_present = [i for i, pred in enumerate(predictions) 
                              if "initial_answer" in pred and pred.get("initial_answer") is not None and 
                              (not isinstance(pred.get("initial_answer"), str) or pred.get("initial_answer").strip() != "")]
    
    if initial_answers_present:
        print(f"Found {len(initial_answers_present)} initial answers to judge...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i in initial_answers_present:
                # Create a temporary prediction object with initial answer as the final answer
                temp_pred = predictions[i].copy()
                temp_pred["final_answer"] = temp_pred["initial_answer"]
                
                future = executor.submit(
                    judge_single_response,
                    temp_pred,
                    gemini_api_key,
                    use_llm,
                    rule_timeout
                )
                future_to_index[future] = i
                
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Judging initial answers"):
                index = future_to_index[future]
                try:
                    judged_initial = future.result()
                    
                    # Copy both the combined judgment and the separate judgments
                    predictions[index]["initial_is_correct"] = judged_initial["is_correct"]
                    predictions[index]["initial_judge_reasoning"] = judged_initial["judge_reasoning"]
                    predictions[index]["initial_rule_is_correct"] = judged_initial["rule_is_correct"]
                    predictions[index]["initial_rule_judge_reasoning"] = judged_initial["rule_judge_reasoning"]
                    if "llm_is_correct" in judged_initial:
                        predictions[index]["initial_llm_is_correct"] = judged_initial["llm_is_correct"]
                        predictions[index]["initial_llm_judge_reasoning"] = judged_initial["llm_judge_reasoning"]
                    
                    # Determine correction outcome
                    final_is_correct = predictions[index].get("is_correct")
                    initial_is_correct = predictions[index].get("initial_is_correct")
                    
                    if initial_is_correct is not None and final_is_correct is not None:
                        if initial_is_correct == False and final_is_correct == True:
                            predictions[index]["correction_outcome"] = "improved"
                        elif initial_is_correct == True and final_is_correct == False:
                            predictions[index]["correction_outcome"] = "worsened"
                        else:
                            predictions[index]["correction_outcome"] = "unchanged"
                    else:
                        predictions[index]["correction_outcome"] = "undetermined"
                        
                except Exception as e:
                    print(f"Error judging initial answer for question {predictions[index].get('qid', 'unknown')}: {e}")
                    predictions[index]["initial_is_correct"] = None
                    predictions[index]["initial_judge_reasoning"] = f"Error during judging: {str(e)}"
                    predictions[index]["correction_outcome"] = "undetermined"
                    # Preserve error field if it exists
                    if "error" in predictions[index]:
                        predictions[index]["error"] = predictions[index]["error"]
    
    return predictions


if __name__ == "__main__":
    model_answer = "(1.00e+16, 2.25e+04, 0.347)"
    correct_answer = "(1 \\times 10^{16}, 2.25 \\times 10^{4}, 0.347)"
    import time
    start_time = time.time()
    print(judge_response_with_rule(model_answer, correct_answer))
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for math_equal: {time_taken:.6f} seconds")    # print(f"Equal: {math_equal(normalized_model, normalized_correct)}")

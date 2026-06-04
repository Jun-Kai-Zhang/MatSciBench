from utils import generate_with_api, extract_final_answer
from utils.image_inputs import entry_images, image_count, image_summary
from methods.prompts import SYSTEM_PROMPT
from collections import Counter

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

def self_consistency(entry, model, max_tokens, temperature, is_multimodal=False, n_samples=4):
    """
    Implements self-consistency method using majority voting.
    Generates multiple responses for the same prompt and selects the most frequent answer.
    """
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
        
        # Use Chain-of-Thought system prompt for better reasoning
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        # Generate multiple responses
        outputs = []
        answers = []
        total_tokens = 0
        
        for _ in range(n_samples):
            full_output, tokens = generate_with_api(
                model,
                conversation,
                max_tokens,
                temperature,
                images
            )
            
            outputs.append(full_output)
            final_answer = extract_final_answer(full_output)
            answers.append(final_answer)
            total_tokens += tokens
        
        # Take majority vote
        if answers:
            answer_counts = Counter(answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            
            # Find index of the majority answer to use its reasoning
            majority_index = answers.index(majority_answer)
            majority_output = outputs[majority_index]
        else:
            majority_answer = ""
            majority_output = ""
        

        return {
            "qid": entry.get("qid", ""),
            "question_type": q_type,
            "question": question_text,
            "full_output": majority_output,
            "final_answer": majority_answer,
            "correct_solution": correct_solution,
            "correct_answer": correct,
            "unit": unit,
            "number_of_answers": number_of_answers,
            "domain": domain,
            "new_token_nums": total_tokens,
            "all_answers": answers,
            "sc_majority": majority_answer,
            "image": image_summary(entry),
            "image_count": image_count(entry),
        }
    
    except Exception as err:
        print(f"Error processing question {entry.get('qid', 'unknown')}: {err}")
        return None

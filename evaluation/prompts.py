JUDGE_SYSTEM_PROMPT = "As an expert judge, evaluate if the following model's answer matches the reference answer. " \
                      "Focus on the numerical values and key concepts. Small numerical differences are tolerable due to approximation errors. " \
                      "For numerical answers, a positive/negative sign mismatch is incorrect unless the reference answer explicitly allows both signs. " \
                      "Don't solve the problem, just judge if the model answer matches the reference answer. " \
                      "Put the final decision ('correct' (if matching) or 'incorrect' (if not matching)) inside a single box using \\boxed{...}. " 

JUDGE_USER_PROMPT = (
    "Question:\n{question}\n\n" \
    "Reference answer:\n{correct_answer}\n\n" \
    "Model answer:\n{model_answer}\n\n" \
    "Is the model answer matching the reference answer? " 
)

    

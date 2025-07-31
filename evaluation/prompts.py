JUDGE_SYSTEM_PROMPT = "As an expert judge, evaluate if the following model's answer matches the reference answer. " \
                      "Focus on the numerical values and key concepts. Small numerical differences are tolerable due to approximation errors. " \
                      "Don't solve the problem, just judge if the model answer matches the reference answer. " \
                      "Put the final decision ('correct' (if matching) or 'incorrect' (if not matching)) inside a single box using \\boxed{...}. " 

JUDGE_USER_PROMPT = (
    "The question is: {question}" \
    "Reference answer: {correct_answer} " \
    "Model answer: {model_answer} " \
    "Is the model answer matching the reference answer? " 
)

    


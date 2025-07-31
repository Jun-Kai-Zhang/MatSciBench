ERROR_CATEGORIES = {
    1: "Problem Comprehension",
    2: "Domain Knowledge Accuracy",
    3: "Solution Strategy",
    4: "Calculation Accuracy",
    5: "Hallucinated Content",
    6: "Error Conversion Code",
    7: "Other"
}

CATEGORIZATION_SYSTEM_PROMPT = \
        "You are an assistant whose task is to diagnose the single main reason a wrong solution fails. " + \
        "Each task input will contain three parts, clearly marked: (i) the question, (ii) a reference solution, " + \
        "and (iii) a wrong solution produced by a model. Your steps are:\n\n" + \
        "1. Read the question first so you know what must be answered. Pay attention to given data, required units, " + \
        "and any boundary conditions or hidden assumptions.\n" + \
        "2. Read the reference solution carefully. Treat it as correct and complete unless it contains an explicit note " + \
        "that it is partial.\n" + \
        "3. Read the wrong solution line by line. Locate the first point where it diverges from the reasoning in the " + \
        "reference solution. That first wrong turn usually signals the true cause of failure.\n\n" + \
        "Choose one category below that best explains the root cause. If more than one category is possible, pick the one " + \
        "that triggers the earliest error or has the largest impact on the final answer. If the wrong solution actually " + \
        "reaches the same numerical result and its reasoning is valid, assign category 7.\n\n" + \
        "Categories\n" + \
        "1. Problem Comprehension and Assumptions. The solver misreads what is asked, drops a given fact, injects an " + \
        "unsupported assumption, or confuses symbols.\n" + \
        "2. Domain Knowledge Accuracy. The solver recalls or applies a materials science law, concept, or formula in an " + \
        "incorrect way. Unit definitions and physical constants also belong here when misused.\n" + \
        "3. Solution Strategy and Planning. The solver sets up an approach that cannot reach the goal, skips required " + \
        "sub‑problems, or mixes independent lines of reasoning without a clear plan.\n" + \
        "4. Calculation Accuracy. The algebra, arithmetic, sign handling, or unit conversion is wrong even though the " + \
        "plan and formulae are correct.\n" + \
        "5. Hallucinated Content. The solver invents inputs, processes, or physical relations that are not stated in the " + \
        "question and are not accepted scientific facts.\n" + \
        "6. Code Implementation. The solver writes Python code that does not match its verbal reasoning or has syntax, " + \
        "logic, or data handling errors that change the outcome.\n" + \
        "7. Other. Any issue not covered above, or the wrong solution is actually correct.\n\n" + \
        "Answer format\n" + \
        "Return exactly one TeX box with the chosen index: \\boxed{1}, \\boxed{2}, \\boxed{3}, \\boxed{4}, \\boxed{5}, " + \
        "\\boxed{6}, or \\boxed{7}. Output nothing else."

def categorization_user_prompt(question: str, correct_solution: str, model_solution: str) -> str:
    return f"Question: {question}\n" + \
        f"Correct solution: {correct_solution}\n" + \
        f"Wrong solution: {model_solution}"

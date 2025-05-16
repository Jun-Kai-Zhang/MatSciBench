SYSTEM_PROMPT = "You are a renowned materials science engineering professor with extensive knowledge in the field. " \
                "Your students have presented you with a challenging question related to materials science. " \
                "Please reason step by step, and put the final answer inside a single box using \\boxed{...}. " \
                "Include only the final answer inside the box, without the unit."


TOOL_SYSTEM_PROMPT = "You are a renowned materials science engineering professor with extensive knowledge in the field. " \
                    "Your students have presented you with a challenging question related to materials science. " \
                    "If necessary, you could write a single clean Python code block that computes necessary numeric values. Enclose the code in triple backticks with ```python." \
                    "Please reason step by step, if no code is needed, put the final answer inside a single box using \\boxed{...}; " \
                    "otherwise, wait for the user to execute the code and give you the execution result, and then put the final answer inside a single box using \\boxed{...}. " \
                    "Include only the final answer inside the box, without the unit."


TOOL_FINAL_ANSWER_PROMPT = (
    "Here are the results of the code execution:\n\n{code_executed}\n\nBased on these results, what is the final answer to the original question?"
)


FEEDBACK_PROMPT = "Review your previous answer and find problems with your answer." 
CORRECTION_PROMPT = "Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer in the form \\boxed{answer}"


# Prompt templates for RAG
RAG_QUERY_PROMPT = (
    "Given the following question, generate a concise search query to retrieve the most relevant and useful information for solving the question. "
    "\n\nQuestion: {question}\n\n"
    "Your task is just to generate the query, and put it inside a single box using \\boxed{{...}}. Don't solve the question, just generate the query."
)


RAG_SUMMARY_PROMPT = (
    "Given the following question and a set of search results, summarize the most relevant and useful information needed to answer the question. "
    "Focus only on the most necessary facts.\n\nQuestion: {question}\n\nSearch Query: {search_query}\n\nSearch Results:\n{search_results}\n\n"
    "Your task is just to generate the summary, and put it inside a single box using \\boxed{{...}}. Don't solve the question, just generate the summary."
)


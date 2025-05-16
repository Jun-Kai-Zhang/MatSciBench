import os
from tavily import TavilyClient
from utils import generate_with_api, extract_final_answer
from methods.prompts import SYSTEM_PROMPT, RAG_QUERY_PROMPT, RAG_SUMMARY_PROMPT

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))




def rag(entry, model, max_tokens, temperature, model_type, llm=None, sampling_params=None, is_multimodal=False):
    """Process a single entry using RAG: generate search query, retrieve, summarize, and answer."""
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
        image_paths = []
        if image_path_raw and image_path_raw.lower() != "nan":
            image_paths = [path.strip() for path in image_path_raw.split(',') if path.strip()]
        if not is_multimodal:
            image_paths = []

        # 1. Generate search query from question
        query_prompt = RAG_QUERY_PROMPT.format(question=question_text)
        query_conversation = [
            {"role": "user", "content": query_prompt}
        ]
        if model_type == "vllm":
            output = llm.chat(query_conversation, sampling_params=sampling_params, use_tqdm=False)
            search_query = output[0].outputs[0].text.strip()
            query_token_nums = len(output[0].outputs[0].token_ids)
        else:
            search_query, query_token_nums = generate_with_api(
                model_type, model, query_conversation, max_tokens, temperature, image_paths
            )
            search_query = search_query.strip()
        # print("full search query: ", search_query)
        search_query = extract_final_answer(search_query)
        # print(f"Search query: {search_query}")

        # 2. Retrieve search results from Tavily
        tavily_response = tavily_client.search(search_query)
        # Concatenate top 3 results for summarization
        search_results = "\n\n".join([
            f"Title: {r['title']}\nContent: {r['content']}" for r in tavily_response.get('results', [])
        ])

        # print(f"Search results: {search_results}")

        # 3. Summarize search results with LLM
        summary_prompt = RAG_SUMMARY_PROMPT.format(question=question_text, search_query=search_query, search_results=search_results)
        summary_conversation = [
            {"role": "user", "content": summary_prompt}
        ]
        if model_type == "vllm":
            output = llm.chat(summary_conversation, sampling_params=sampling_params, use_tqdm=False)
            summary = output[0].outputs[0].text.strip()
            summary_token_nums = len(output[0].outputs[0].token_ids)
        else:
            summary, summary_token_nums = generate_with_api(
                model_type, model, summary_conversation, max_tokens, temperature, image_paths
            )
            summary = summary.strip()
        # print("full summary: ", summary)
        summary = extract_final_answer(summary)
        # print("extracted summary: ", summary)
        # 4. Append summary to question
        augmented_question = question_text + "\n\n[Relevant Information from Search:]\n" + "search query: " + search_query + "\n\n" + "summarized information: " + summary

        # print(f"Augmented question: {augmented_question}")

        # 5. Answer the question with LLM
        answer_conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_question}
        ]

        if model_type == "vllm":
            output = llm.chat(answer_conversation, sampling_params=sampling_params, use_tqdm=False)
            full_output = output[0].outputs[0].text.strip()
            answer_token_nums = len(output[0].outputs[0].token_ids)
        else:
            full_output, answer_token_nums = generate_with_api(
                model_type, model, answer_conversation, max_tokens, temperature, image_paths
            )
        final_answer = extract_final_answer(full_output) if full_output else ""
        new_token_nums = query_token_nums + summary_token_nums + answer_token_nums

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
            "error": None
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
            "error": str(err)
        }

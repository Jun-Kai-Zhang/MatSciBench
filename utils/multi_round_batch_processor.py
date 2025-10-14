"""
Multi-round batch processor for tool augmentation compatibility.
Implements iterative batch processing where first round generates code,
and second round processes the code execution results.
"""

import json
import os
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import openai
import anthropic
import re
from utils.python_executor import PythonExecutor
from methods.prompts import TOOL_SYSTEM_PROMPT, TOOL_FINAL_ANSWER_PROMPT
from utils.batch_processor import (
    create_openai_batch_file, create_claude_batch_requests,
    wait_for_openai_batch, wait_for_claude_batch,
    submit_openai_batch, submit_claude_batch,
    download_openai_results, download_claude_results
)


class MultiRoundBatchProcessor:
    """Handles multi-round batch processing for tool augmentation."""

    def __init__(self, model: str, max_tokens: int, temperature: float, model_type: str, is_multimodal: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_type = model_type
        self.is_multimodal = is_multimodal
        self.openai_client = openai.OpenAI() if model_type == "openai" else None
        self.anthropic_client = anthropic.Anthropic() if model_type == "anthropic" else None

    def extract_code_blocks(self, full_output: str) -> List[str]:
        """Extract Python code blocks from the model output."""
        return re.findall(r"```python(.*?)```", full_output, re.DOTALL)

    def execute_code(self, code_str: str, timeout: int = 30) -> str:
        """Execute the extracted code using PythonExecutor and return the output."""
        executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=timeout)
        result, report = executor.apply(code_str)
        return result

    def prepare_round1_prompt(self, entry: Dict) -> Dict:
        """Prepare the initial prompt for tool augmentation."""
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

    def prepare_round2_prompt(self, entry: Dict, round1_response: str, code_executed: str) -> Dict:
        """Prepare the follow-up prompt with code execution results."""
        # Recreate the initial conversation
        round1_prompt = self.prepare_round1_prompt(entry)
        conversation = round1_prompt["messages"].copy()

        # Add the model's response from round 1
        conversation.append({"role": "assistant", "content": round1_response})

        # Add the code execution results and request final answer
        conversation.append({
            "role": "user",
            "content": TOOL_FINAL_ANSWER_PROMPT.format(code_executed=code_executed)
        })

        return {"messages": conversation}

    def process_round1_results(self, batch_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Process round 1 results and separate entries that need round 2."""
        no_code_entries = []  # Entries that don't need code execution
        need_round2_entries = []  # Entries that need code execution and round 2

        for result in batch_results:
            entry = result["entry"]
            response = result["response"]

            # Extract code blocks from the response
            code_blocks = self.extract_code_blocks(response)

            if not code_blocks:
                # No code found, this entry is complete
                from utils import extract_final_answer
                final_answer = extract_final_answer(response)

                no_code_entries.append({
                    "qid": entry.get("qid", "unknown"),
                    "question_type": entry["type"],
                    "question": entry["question"],
                    "model_output": response,
                    "code_execution_result": "",
                    "final_answer": final_answer,
                    "correct_answer": str(entry["answer"]).strip() if entry["answer"] is not None else "",
                    "new_token_nums": result.get("token_count", 0)
                })
            else:
                # Execute all code blocks and combine their results
                code_executed = ""
                for i, code_block in enumerate(code_blocks):
                    block_result = self.execute_code(code_block)
                    code_executed += f"Code block {i+1} execution result:\n{block_result}\n\n"

                # Prepare for round 2
                need_round2_entries.append({
                    "entry": entry,
                    "round1_response": response,
                    "code_executed": code_executed,
                    "round1_tokens": result.get("token_count", 0)
                })

        return no_code_entries, need_round2_entries

    def process_round2_results(self, round2_entries: List[Dict], batch_results: List[Dict]) -> List[Dict]:
        """Process round 2 results and create final outputs."""
        final_results = []

        for i, result in enumerate(batch_results):
            round2_entry = round2_entries[i]
            entry = round2_entry["entry"]
            round1_response = round2_entry["round1_response"]
            code_executed = round2_entry["code_executed"]
            followup_response = result["response"]

            from utils import extract_final_answer
            final_answer = extract_final_answer(followup_response)

            # Combine outputs
            full_output = round1_response
            full_output += "\n\n" + "-" * 60
            full_output += "\n\n" + code_executed
            full_output += "\n\n" + "-" * 60
            full_output += f"\n\n{followup_response}"

            final_results.append({
                "qid": entry.get("qid", "unknown"),
                "question_type": entry["type"],
                "question": entry["question"],
                "model_output": full_output,
                "code_execution_result": code_executed,
                "final_answer": final_answer,
                "correct_answer": str(entry["answer"]).strip() if entry["answer"] is not None else "",
                "new_token_nums": round2_entry["round1_tokens"] + result.get("token_count", 0)
            })

        return final_results

    def process_batch_tool_augmentation(self, data: List[Dict]) -> List[Dict]:
        """Main function to process tool augmentation using multi-round batch processing."""
        print(f"Starting multi-round batch processing for {len(data)} entries...")

        # Round 1: Generate initial responses
        print("=== ROUND 1: Generating initial responses ===")

        if self.model_type == "openai":
            # Create batch file manually instead of using the generic function
            batch_requests = []
            for i, entry in enumerate(data):
                # Prepare the prompt using our method
                prompt_result = self.prepare_round1_prompt(entry)
                messages = prompt_result.get('messages', [])

                # Create custom_id using the same logic as batch_processor.py
                clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
                if len(clean_qid) > 40:  # Leave room for prefix
                    clean_qid = clean_qid[:40]
                custom_id = f"req_{i}_{clean_qid}"

                # Create the request with appropriate parameters for the model
                request_body = {
                    "model": self.model,
                    "messages": messages
                }

                # Use max_completion_tokens for GPT-5, o1, o3, o4 models, max_tokens for others
                if self.model.startswith("gpt-5") or self.model.startswith("o1-") or self.model.startswith("o3-") or self.model.startswith("o4-"):
                    request_body["max_completion_tokens"] = self.max_tokens
                    if self.model.startswith("o4-"):
                        request_body["temperature"] = self.temperature
                else:
                    request_body["max_tokens"] = self.max_tokens
                    request_body["temperature"] = self.temperature

                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body
                }
                batch_requests.append(batch_request)

            # Write batch file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + '\n')
                batch_file_path = f.name

            # Submit and wait for batch completion
            batch_id = submit_openai_batch(batch_file_path, f"MatSciBench Tool Augmentation Round 1 - {self.model}")
            print(f"Round 1 batch submitted: {batch_id}")
            batch_obj = wait_for_openai_batch(batch_id)

            # Download and process results using the existing function
            batch_results = download_openai_results(batch_obj, data)

            # Convert to expected format for our processor
            round1_results = []
            for result in batch_results:
                round1_results.append({
                    "entry": next((item for item in data if item.get('qid') == result['qid']), {}),
                    "response": result['full_output'],
                    "token_count": result['new_token_nums']
                })

        elif self.model_type == "anthropic":
            # Create batch requests manually for Anthropic
            batch_requests = []
            for i, entry in enumerate(data):
                # Prepare the prompt using our method
                prompt_result = self.prepare_round1_prompt(entry)
                messages = prompt_result.get('messages', [])

                # Create custom_id using the same logic as batch_processor.py
                clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', i)))
                if len(clean_qid) > 40:  # Leave room for prefix
                    clean_qid = clean_qid[:40]
                custom_id = f"req_{i}_{clean_qid}"

                # Create the request for Anthropic batch API
                batch_request = {
                    "custom_id": custom_id,
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "messages": messages
                    }
                }
                if self.temperature != 1.0:  # Only add temperature if not default
                    batch_request["params"]["temperature"] = self.temperature

                batch_requests.append(batch_request)

            # Submit and wait for batch completion
            batch_id = submit_claude_batch(batch_requests, f"MatSciBench Tool Augmentation Round 1 - {self.model}")
            print(f"Round 1 batch submitted: {batch_id}")
            batch_obj = wait_for_claude_batch(batch_id)

            # Download and process results using the existing function
            batch_results = download_claude_results(batch_obj, data)

            # Convert to expected format for our processor
            round1_results = []
            for result in batch_results:
                round1_results.append({
                    "entry": next((item for item in data if item.get('qid') == result['qid']), {}),
                    "response": result['full_output'],
                    "token_count": result['new_token_nums']
                })

        else:
            raise ValueError(f"Unsupported model type for batch processing: {self.model_type}")

        # Process round 1 results
        print("=== Processing Round 1 Results ===")
        completed_entries, need_round2_entries = self.process_round1_results(round1_results)

        print(f"Round 1 complete: {len(completed_entries)} entries finished, {len(need_round2_entries)} need Round 2")

        # Round 2: Process entries that need code execution follow-up
        if need_round2_entries:
            print("=== ROUND 2: Processing code execution follow-ups ===")

            # Prepare round 2 data
            round2_data = []
            for entry_data in need_round2_entries:
                entry = entry_data["entry"]
                round2_prompt = self.prepare_round2_prompt(
                    entry,
                    entry_data["round1_response"],
                    entry_data["code_executed"]
                )
                round2_data.append({"entry": entry, "prompt": round2_prompt})

            if self.model_type == "openai":
                # Create batch file manually for round 2
                batch_requests = []
                for i, item in enumerate(round2_data):
                    messages = item["prompt"]["messages"]
                    entry = item["entry"]

                    # Create custom_id using the same logic as batch_processor.py
                    clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', f'round2_{i}')))
                    if len(clean_qid) > 40:  # Leave room for prefix
                        clean_qid = clean_qid[:40]
                    custom_id = f"req_{i}_{clean_qid}"

                    # Create the request with appropriate parameters for the model
                    request_body = {
                        "model": self.model,
                        "messages": messages
                    }

                    # Use max_completion_tokens for GPT-5, o1, o3, o4 models, max_tokens for others
                    if self.model.startswith("gpt-5") or self.model.startswith("o1-") or self.model.startswith("o3-") or self.model.startswith("o3-") or self.model.startswith("o4-"):
                        request_body["max_completion_tokens"] = self.max_tokens
                        if self.model.startswith("o4-"):
                            request_body["temperature"] = self.temperature
                    else:
                        request_body["max_tokens"] = self.max_tokens
                        request_body["temperature"] = self.temperature

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body
                    }
                    batch_requests.append(batch_request)

                # Write batch file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                    for request in batch_requests:
                        f.write(json.dumps(request) + '\n')
                    batch_file_path = f.name

                batch_id = submit_openai_batch(batch_file_path, f"MatSciBench Tool Augmentation Round 2 - {self.model}")
                print(f"Round 2 batch submitted: {batch_id}")
                batch_obj = wait_for_openai_batch(batch_id)

                # Create data for download function using actual entries
                round2_dummy_data = []
                for i, item in enumerate(round2_data):
                    round2_dummy_data.append(item['entry'])

                # Download and process results using the existing function
                batch_results = download_openai_results(batch_obj, round2_dummy_data)

                # Convert to expected format for our processor
                round2_results = []
                for result in batch_results:
                    round2_results.append({
                        "response": result['full_output'],
                        "token_count": result['new_token_nums']
                    })

            elif self.model_type == "anthropic":
                # Create batch requests manually for round 2
                batch_requests = []
                for i, item in enumerate(round2_data):
                    messages = item["prompt"]["messages"]
                    entry = item["entry"]

                    # Create custom_id using the same logic as batch_processor.py
                    clean_qid = "".join(c if c.isalnum() or c in '_-' else '_' for c in str(entry.get('qid', f'round2_{i}')))
                    if len(clean_qid) > 40:  # Leave room for prefix
                        clean_qid = clean_qid[:40]
                    custom_id = f"req_{i}_{clean_qid}"

                    # Create the request for Anthropic batch API
                    batch_request = {
                        "custom_id": custom_id,
                        "params": {
                            "model": self.model,
                            "max_tokens": self.max_tokens,
                            "messages": messages
                        }
                    }
                    if self.temperature != 1.0:  # Only add temperature if not default
                        batch_request["params"]["temperature"] = self.temperature

                    batch_requests.append(batch_request)

                batch_id = submit_claude_batch(batch_requests, f"MatSciBench Tool Augmentation Round 2 - {self.model}")
                print(f"Round 2 batch submitted: {batch_id}")
                batch_obj = wait_for_claude_batch(batch_id)

                # Create data for download function using actual entries
                round2_dummy_data = []
                for i, item in enumerate(round2_data):
                    round2_dummy_data.append(item['entry'])

                # Download and process results using the existing function
                batch_results = download_claude_results(batch_obj, round2_dummy_data)

                # Convert to expected format for our processor
                round2_results = []
                for result in batch_results:
                    round2_results.append({
                        "response": result['full_output'],
                        "token_count": result['new_token_nums']
                    })

            # Process round 2 results
            round2_completed = self.process_round2_results(need_round2_entries, round2_results)
            completed_entries.extend(round2_completed)

        print(f"Multi-round batch processing complete: {len(completed_entries)} total entries processed")
        return completed_entries


def run_multi_round_batch_tool_augmentation(data: List[Dict], model: str, max_tokens: int,
                                           temperature: float, model_type: str,
                                           is_multimodal: bool = False) -> List[Dict]:
    """
    Main entry point for multi-round batch tool augmentation processing.

    Args:
        data: List of question entries
        model: Model name
        max_tokens: Maximum tokens per request
        temperature: Sampling temperature
        model_type: Type of model ("openai", "anthropic")
        is_multimodal: Whether the model supports images

    Returns:
        List of processed results with final answers
    """
    processor = MultiRoundBatchProcessor(model, max_tokens, temperature, model_type, is_multimodal)
    return processor.process_batch_tool_augmentation(data)
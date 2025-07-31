from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import json
import os
import concurrent.futures
from tqdm import tqdm
import argparse
import re
import openai
from openai import OpenAI


class QA(BaseModel):
    question: str
    question_image: str 
    solution: str
    solution_image: str
    answer_type: str 
    number_of_answers: str  
    answer: str
    unit: str
    
def load_examples_from_csv(csv_path, max_examples=3):
    """Load example QA pairs from a CSV file to use as in-context examples."""
    try:
        examples_df = pd.read_csv(csv_path)
        examples = []
        
        # Take a small sample of examples to include in the prompt
        for _, row in examples_df.head(max_examples).iterrows():
            example = {
                "question": row.get("question", ""),
                "solution": row.get("solution", ""),
                "type": row.get("type", ""),
                "number_of_answers": row.get("number_of_answers", ""),
                "answer": row.get("answer", ""),
                "unit": row.get("unit", ""),
                "image": row.get("image", "")
                                        }
            examples.append(example)
        
        return examples
    except Exception as e:
        print(f"Error loading examples: {e}")
        return []

def format_examples(examples):
    """Format examples as text for inclusion in the prompt."""
    formatted_examples = ""
    
    for i, example in enumerate(examples, 1):
        formatted_examples += f"""
            Example {i}:

            {{
            "question": "{example['question']}",
            "solution": "{example['solution']}",
            "answer_type": "{example['type']}",
            "number_of_answers": "{example['number_of_answers']}",
            "answer": "{example['answer']}",
            "unit": "{example['unit']}"
            }}
            
            """
    return formatted_examples

def revise_problem_content(problem, solution, reference, model_provider="gemini"):
    """First stage: Use the selected model to revise and ensure completeness of problem and solution."""
    
    prompt = f"""
    Review this math problem and solution:
    
    Question: {problem}
    Solution: {solution}
    Reference Context: {reference}
    
    Please:
    1. Revise the question if needed to ensure accuracy and completeness with the reference context. Don't change the phrasing of the question.
    2. Revise the solution if needed to ensure accuracy and completeness with the reference context. Don't change the phrasing of the solution.
    3. Make sure all fields are delivered in proper latex format if necessary.
    
    Return your analysis in JSON format with these fields:
    - question: The accurate and complete question text
    - solution: The accurate and complete solution text
    """
    
    try:
        if model_provider.lower() == "gemini":
            # Configure the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Generate content with the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Extract response text safely
            response_text = ""
            try:
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        response_text = content.parts[0].text
            except Exception as extract_err:
                print(f"Error extracting response text: {extract_err}")
                return {
                    "question": problem,
                    "solution": solution
                }
        
        elif model_provider.lower() == "deepseek":
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            
            # Generate content with the DeepSeek model
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that revises math problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # If we couldn't get a valid response
        if not response_text:
            print(f"Received empty response from {model_provider}")
            return {
                "question": problem,
                "solution": solution
            }
        
        # Parse JSON response with improved error handling
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {json_err}")
            # Use our new safe parsing function with appropriate defaults
            return safe_json_parse(response_text, {
                "question": problem,
                "solution": solution
            })
            
    except Exception as e:
        print(f"Error in revision step with {model_provider}: {e}")
        # Return original content
        return {
            "question": problem,
            "solution": solution
        }

def extract_references(question, solution, reference, model_provider="gemini"):
    """Second stage: Retune question and solution, and extract referenced images, tables and equations."""
    
    prompt = f"""
    Review this math problem, solution, and reference context:
    
    Question: {question}
    Solution: {solution}
    Reference Context: {reference}
    
    Please:
    1. Check the question and solution on accuracy and completeness based on the reference context.
    2. If the question or solution statement refers to a figure, then find the figure by the Figure index and include its file path in question_image or solution_image field. 
    Otherwise, leave the question_image or solution_image field empty. Don't include a figure that is not referred in the question or solution statement.
    3. If there are multiple images, separate them with commas.
    4. If the question or solution statement refers to a table, then find the table and append the information to the question or solution statement.
    5. If the question or solution statement refers to an equation by the Equation index, then find the equation if you can and add the equation to the question or solution statement.
    6. Make sure all fields are delivered in proper latex format if necessary.
    
    Return your analysis in JSON format with these fields:
    - question: The accurate and complete question text
    - solution: The accurate and complete solution text
    - question_image: The path(s) to any images referenced in the question
    - solution_image: The path(s) to any images referenced in the solution
    """
    
    try:
        if model_provider.lower() == "gemini":
            # Configure the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Generate content with the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Extract response text safely
            response_text = ""
            try:
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        response_text = content.parts[0].text
            except Exception as extract_err:
                print(f"Error extracting response text: {extract_err}")
                return {
                    "question_image": "",
                    "solution_image": ""
                }
        
        elif model_provider.lower() == "deepseek":
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            
            # Generate content with the DeepSeek model
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts references from math problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # If we couldn't get a valid response
        if not response_text:
            print(f"Received empty response from {model_provider}")
            return {
                "question_image": "",
                "solution_image": ""
            }
        
        # Parse JSON response with improved error handling
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {json_err}")
            return safe_json_parse(response_text, {
                "question": question,
                "solution": solution,
                "question_image": "",
                "solution_image": ""
            })
            
    except Exception as e:
        print(f"Error in reference extraction with {model_provider}: {e}")
        return {
            "question_image": "",
            "solution_image": ""
        }

def process_problem_content(problem, solution, reference, model_provider="gemini"):
    """Combined two-stage approach: first revise content, then extract references and further refine."""
    
    # First stage: Revise the content
    revised_content = revise_problem_content(problem, solution, reference, model_provider)
    
    # Extract the revised question and solution
    revised_question = revised_content.get('question', problem)
    revised_solution = revised_content.get('solution', solution)
    
    # Second stage: Extract references and further refine the content
    references = extract_references(revised_question, revised_solution, reference, model_provider)
    
    # Combine the results, preferring the retuned question and solution from extract_references
    result = {
        "question": references.get('question', revised_question),
        "solution": references.get('solution', revised_solution),
        "question_image": references.get('question_image', ''),
        "solution_image": references.get('solution_image', '')
    }
    
    return result

def process_problem_fields(question, solution, model_provider="gemini"):
    """Second stage: Determine answer type, number of answers, extract answer and unit."""
    
    # Load examples from CSV
    examples = load_examples_from_csv("preprocess/example_qa.csv")
    formatted_examples = format_examples(examples)
    
    prompt = f"""
    For this math problem and solution:
    
    Question: {question}
    Solution: {solution}
    
    Please determine the following:
    1. Determine if the answer type is a number (NUM), formula (FORMULA), or multiple choice (MCQ) and fill the answer_type field accordingly.
    2. Determine if there are "single" or "multiple" answers required and fill the number_of_answers field accordingly.
    3. Extract the final answer(s) from the solution (the number part if there is unit) and fill the answer field accordingly.
    4. Extract any units for the answer if applicable (e.g., m, kg, m/s, etc.) and fill the unit field accordingly. Otherwise, leave the unit field empty.
    5. Make sure all fields are delivered in proper latex format if necessary.

    Return your analysis in JSON format with these fields:
    - answer_type: The type of answer (NUM, FORMULA, or MCQ)
    - number_of_answers: Whether the problem requires "single" or "multiple" answers
    - answer: The extracted answer(s) from the solution
    - unit: Any units for the answer, if applicable
    
    Some examples:
    {formatted_examples}
    """
    
    try:
        if model_provider.lower() == "gemini":
            # Configure the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Generate content with the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Extract response text
            if not hasattr(response, 'text'):
                if hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        response_text = content.parts[0].text
                    else:
                        print(f"Response structure: {dir(response)}")
                        raise Exception("Unable to extract text from response")
                else:
                    raise Exception("No candidates found in response")
            else:
                response_text = response.text
                
        elif model_provider.lower() == "deepseek":
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            
            # Generate content with the DeepSeek model
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes math problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Parse JSON response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {json_err}")
            # Try to clean the response before retrying
            cleaned_text = response_text.strip()
            # Remove common problematic escape sequences
            cleaned_text = cleaned_text.replace('\\(', '(').replace('\\)', ')').replace('\\[', '[').replace('\\]', ']')
            # Find where the JSON object likely ends
            json_end = cleaned_text.rfind('}')
            if json_end > 0:
                cleaned_text = cleaned_text[:json_end+1]
            
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Try to extract fields using regex if JSON parsing fails
                result = {}
                for field in ["answer_type", "number_of_answers", "answer", "unit"]:
                    field_match = re.search(f'"{field}"\\s*:\\s*"([^"]*)"', response_text)
                    if field_match:
                        result[field] = field_match.group(1)
                    else:
                        # Use default values
                        if field == "answer_type":
                            result[field] = "NUM"  # default
                        elif field == "number_of_answers":
                            result[field] = "single"  # default
                        elif field == "unit":
                            result[field] = ""  # default empty string
                        elif field == "answer":
                            result[field] = "Not extracted"
                
                return result
            
    except Exception as e:
        print(f"Error in second {model_provider} API call: {e}")
        # Return default values
        return {
            "answer_type": "NUM",  # default
            "number_of_answers": "single",  # default
            "answer": "Not extracted", 
            "unit": ""
        }

def rephrase_multiple_answer_question(question, answer, unit, model_provider="deepseek"):
    """Use the selected model to rephrase questions with multiple answers."""
    
    prompt = f"""
    For this math problem:
    
    Question: {question}
    Current Answer: {answer}
    Current Unit: {unit}
    
    This question requires multiple answers.
    
    1. Analyze the question to determine what specific values are being asked for.
    2. If there is part of the question that is not asking for a number/formula/choice, then remove it. (for example, if the question ask you to explain something, then remove it)
    3. Add a sentence at the end of the question that says "Give your answer as a tuple (X, Y, ...)" where X, Y, ... are replaced with clear descriptions of what each value in the answer represents.
    4. For example, if the question asks for both the area and perimeter, the added sentence would be "Give your answer as an ordered tuple (area, perimeter)."
    
    Return your analysis in JSON format with these fields:
    - revised_question: The original question with the added instruction
    - revised_answer: The answer formatted properly as a tuple if needed. Render each answer in the latex format if necessary.
    - revised_unit: The unit formatted as a tuple. If some unit is not applicable, leave it empty in the tuple. If unit is not needed at all, then leave it blank.
    """
    
    try:
        if model_provider.lower() == "gemini":
            # Configure the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Generate content with the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Extract response text
            if not hasattr(response, 'text'):
                if hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        response_text = content.parts[0].text
                    else:
                        raise Exception("Unable to extract text from response")
                else:
                    raise Exception("No candidates found in response")
            else:
                response_text = response.text
                
        elif model_provider.lower() == "deepseek":
            # Use the OpenAI client for DeepSeek
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            
            # Generate content with the DeepSeek model
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases math problems with multiple answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Handle JSON parsing with better error recovery
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error in rephrasing: {json_err}")
            print(f"Raw response: {response_text[:200]}...")
            
            # Multiple fallback parsing strategies
            # 1. Fix common JSON escaping issues and try again
            cleaned_text = response_text.replace('\\(', '(').replace('\\)', ')').replace('\\[', '[').replace('\\]', ']')
            cleaned_text = re.sub(r'\\([^\\"])', r'\1', cleaned_text)  # Fix invalid escapes
            
            # 2. Trim to find a valid JSON object
            if cleaned_text.find('{') >= 0 and cleaned_text.rfind('}') > cleaned_text.find('{'):
                start = cleaned_text.find('{')
                end = cleaned_text.rfind('}') + 1
                cleaned_text = cleaned_text[start:end]
            
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # 3. Extract fields using regex as last resort
                result = {}
                fields_to_extract = [
                    ("revised_question", question),
                    ("revised_answer", answer),
                    ("revised_unit", unit)
                ]
                
                for field_name, default_value in fields_to_extract:
                    # Try with double quotes first
                    pattern = f'"{field_name}"\\s*:\\s*"(.*?)"(?=,|\\s*}})'
                    match = re.search(pattern, cleaned_text, re.DOTALL)
                    
                    if not match:
                        # Try with different quote patterns
                        pattern = f'"{field_name}"\\s*:\\s*\'(.*?)\'(?=,|\\s*}})'
                        match = re.search(pattern, cleaned_text, re.DOTALL)
                    
                    if not match:
                        # Try without quotes around the value
                        pattern = f'"{field_name}"\\s*:\\s*(.*?)(?=,\\s*"|\\s*}})'
                        match = re.search(pattern, cleaned_text, re.DOTALL)
                    
                    result[field_name] = match.group(1).strip() if match else default_value
                
                return result
    except Exception as e:
        print(f"Error in rephrasing with {model_provider}: {e}")
        # Return the original values if there's an error
        return {
            "revised_question": question,
            "revised_answer": answer,
            "revised_unit": unit
        }

# Add a helper function to improve JSON parsing throughout the code
def safe_json_parse(response_text, default_values=None):
    """Safely parse JSON with multiple fallback strategies."""
    if default_values is None:
        default_values = {}
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Clean up the response text
        cleaned_text = response_text.strip()
        # Fix common escape issues
        cleaned_text = cleaned_text.replace('\\(', '(').replace('\\)', ')').replace('\\[', '[').replace('\\]', ']')
        cleaned_text = re.sub(r'\\([^\\"])', r'\1', cleaned_text)
        
        # Try to find a valid JSON object
        if cleaned_text.find('{') >= 0 and cleaned_text.rfind('}') > cleaned_text.find('{'):
            start = cleaned_text.find('{')
            end = cleaned_text.rfind('}') + 1
            cleaned_text = cleaned_text[start:end]
            
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        # Last resort: extract fields with regex
        result = default_values.copy()
        for field in default_values.keys():
            field_match = re.search(f'"{field}"\\s*:\\s*"(.*?)"(?=,|\\s*}})', cleaned_text, re.DOTALL)
            if field_match:
                result[field] = field_match.group(1)
            else:
                # Try without quotes
                field_match = re.search(f'"{field}"\\s*:\\s*(.*?)(?=,\\s*"|\\s*}})', cleaned_text, re.DOTALL)
                if field_match:
                    result[field] = field_match.group(1).strip()
        
        return result

def process_single_row(row, model_provider="gemini"):
    """Process a single row of data using the two-stage approach."""
    try:
        # Stage 1: Process content completeness and images
        content_result = process_problem_content(
            row['problem'], 
            row['solution'], 
            row['reference'],
            model_provider
        )
        
        # Extract revised content
        revised_question = content_result.get('question', row['problem'])
        revised_solution = content_result.get('solution', row['solution'])
        question_image = content_result.get('question_image', '')
        solution_image = content_result.get('solution_image', '')
        
        # Convert dictionary solutions to strings if needed
        if isinstance(revised_solution, dict):
            # Convert the solution dictionary to a formatted string
            solution_string = ""
            for key, value in revised_solution.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    solution_string += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        solution_string += f"  {sub_key}: {sub_value}\n"
                else:
                    solution_string += f"{key}: {value}\n"
            revised_solution = solution_string.strip()
        
        # Stage 2: Process answer fields
        fields_result = process_problem_fields(
            revised_question,
            revised_solution,
            model_provider
        )
        
        # Ensure answer and unit are strings, not lists
        answer = fields_result.get('answer', 'Not extracted')
        if isinstance(answer, list):
            answer = ', '.join(str(item) for item in answer)
            
        unit = fields_result.get('unit', '')
        if isinstance(unit, list):
            unit = ', '.join(str(item) for item in unit)
        elif unit is None:
            unit = ''
            
        # Combine results from both stages
        result_dict = {
            "example_id": row['example_id'],
            "question": revised_question,
            "question_image": question_image,
            "solution": revised_solution,
            "solution_image": solution_image,
            "answer_type": fields_result.get('answer_type', 'NUM'),
            "number_of_answers": fields_result.get('number_of_answers', 'single'),
            "answer": answer,
            "unit": unit
        }
        
        # Create a QA object to validate the structure
        qa = QA(**result_dict)
        result_dict = qa.model_dump()
        result_dict['example_id'] = row['example_id']
        
        # Rephrase for multiple answers
        if result_dict['number_of_answers'].lower() == 'multiple':
            try:
                rephrase_result = rephrase_multiple_answer_question(
                    result_dict['question'],
                    result_dict['answer'],
                    result_dict['unit'],
                    model_provider
                )
                result_dict['question'] = rephrase_result.get('revised_question', result_dict['question'])
                result_dict['answer'] = rephrase_result.get('revised_answer', result_dict['answer'])
                result_dict['unit'] = rephrase_result.get('revised_unit', result_dict['unit'])
            except Exception as e:
                print(f"Error in rephrasing for example {row['example_id']}: {e}")
                # Continue with original values
        
        return result_dict
    except Exception as e:
        print(f"Error processing row {row['example_id']}: {e}")
        # Return basic data with defaults in case of error
        return {
            "example_id": row['example_id'],
            "question": row['problem'],
            "question_image": "",
            "solution": row['solution'],
            "solution_image": "",
            "answer_type": "NUM",
            "number_of_answers": "single",
            "answer": "Not extracted",
            "unit": ""
        }

def process_csv(csv_path, output_path=None, max_workers=5, model_provider="gemini"):
    """Process the CSV file in parallel and optionally save results to a new CSV."""
    df = pd.read_csv(csv_path)
    results = []
    
    # Create a list of row dictionaries for processing
    rows = df.to_dict('records')
    
    # Process rows in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show a progress bar
        futures = {executor.submit(process_single_row, row, model_provider): row for row in rows}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows), desc="Processing rows"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    if output_path:
        pd.DataFrame(results).to_csv(output_path, index=False)
    
    return results

def ensure_proper_latex_formatting(qa_dict, model_provider="gemini"):
    """Final check: Ensure all fields have proper LaTeX syntax and math symbols are wrapped in math environment."""
    
    prompt = f"""
    Recheck these math problem fields and ensure proper LaTeX formatting and satisfy other format requirements:
    
    Question: {qa_dict['question']}
    Solution: {qa_dict['solution']}
    Answer: {qa_dict['answer']}
    Unit: {qa_dict.get('unit', '')}
    
    Please:
    1. Check for any mathematical expressions or symbols that should be wrapped in LaTeX math delimiters but aren't.
    2. Ensure all equations, fractions, subscripts, superscripts, and special symbols are properly formatted with LaTeX syntax.
    3. Use inline math mode ($ ... $) for simple expressions within text.
    4. Use display math mode ($$ ... $$ or \\[ ... \\]) for standalone equations or complex expressions.
    5. Make sure units have proper LaTeX formatting if needed.
    6. Do not change the meaning or content of any field, just fix the LaTeX formatting.
    7. Both answer and unit fields are returned as a tuple (X, Y, ...) if there are multiple answers or units. Don't add unnecessary double quotes around the answers. 
    8. If the question requires multiple answers, check if the last sentence in question specifing what each answer represents is correct. If not, revise it.
    9. Check whether this answer extracted from the solution is correct. If not, revise it.
    10. Make sure the answer corresponds to the last sentence in question specifing what each answer represents.
    11. If there is no unit, then return "" in the unit field. Don't return empty () in the unit field.
    
    Return your analysis in JSON format with these fields:
    - question: The question with proper LaTeX formatting
    - solution: The solution with proper LaTeX formatting 
    - answer: The answer with proper LaTeX formatting
    - unit: The unit with proper LaTeX formatting
    """
    
    try:
        if model_provider.lower() == "gemini":
            # Configure the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Generate content with the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Extract response text safely
            response_text = ""
            try:
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        response_text = content.parts[0].text
            except Exception as extract_err:
                print(f"Error extracting response text: {extract_err}")
                return qa_dict
        
        elif model_provider.lower() == "deepseek":
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            
            # Generate content with the DeepSeek model
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ensures proper LaTeX formatting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            # Update only the specified fields, preserving other fields
            for field in ["question", "solution", "answer", "unit"]:
                if field in result and result[field]:
                    qa_dict[field] = result[field]
            
            return qa_dict
            
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error in LaTeX formatting: {json_err}")
            return safe_json_parse(response_text, qa_dict)
            
    except Exception as e:
        print(f"Error in LaTeX formatting check with {model_provider}: {e}")
        return qa_dict

# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process math problems with AI models and extract information.')
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', default='preprocess/revised_qa', required=False, help='Path to save the output CSV file')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of parallel workers (default: 8)')
    parser.add_argument('--model_provider', choices=['gemini', 'deepseek'], default='deepseek', 
                        help='Model provider to use for processing (default: gemini)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the arguments
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(args.input_csv))[0]
    # Add model provider to the filename
    output_filename = f"{base_filename}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    processed_data = process_csv(args.input_csv, output_path, max_workers=args.max_workers, 
                                model_provider=args.model_provider)

    print(f"Processed {len(processed_data)} problems using {args.model_provider} and saved to {output_path}")



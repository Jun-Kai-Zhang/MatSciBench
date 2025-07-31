# MatSciBench

## Usage

To launch the evaluation, just run

```bash
python evaluation/eval.py --method ['base', 'tool', or 'correction'] --llm_judge [or --rule_judge]
```

## Data Preprocessing
First use the mistral api to ocr the textbook and save the ocr result in the preprocess/textbook_ocr folder.
```
python preprocess/ocr.py --textbook [the file path of the pdf format textbook]
```

Then extract the question answer pairs from the ocr result and save the extracted result in a csv file in the preprocess/extracted_qa folder.
```
python preprocess/extract.py --directory [the directory of the ocr result]
```

Next, use the deepseek/gemini api to revise the ocr result and save the revised result in a csv file in the preprocess/revised_qa folder.
```
python preprocess/revise.py --input_csv [the path of the extracted qa csv file]
```

Finally, do necessary formatting and save the final dataset in the datasets folder.
```
python preprocess/finalize.py --csv [the path of the revised qa csv file]
```

However, after this process, some common issues still exist, please refer to the Common Issues in Extracted QA pairs section for more details.

## Criteria for Questions
1. All math symbols should be rendered using LaTeX.
2. Exclude questions that is hard to evaluate, answer is some figures, exclude question that is not self-contained (etc. having ambiguous symbols).
3. Feel free to delete the question, rephrase the question, add more details, or only keep part of the question.
4. There are 3 types of questions: NUM, FORMULA, MCQ. which are numerical answers, formula answers, and multiple choice answers. If the question don't fit any of the types, feel free to edit the question so that it fits one of the types. The value tupe is used when the question is asking for multiple values. 
5. A question can have a single answer or multiple answers. If there are multiple answers, please return the answer as a tuple. In this case, we should also add an instruction to the end of the question, e.g., "Give your answer as a tuple ($\sigma_2 / \sigma_1$ for zinc, $\sigma_2 / \sigma_1$ for steel).".
6. Please separately put the unit in the unit field. If there are multiple units, please return the unit as a tuple.

## Guide on data editing UI tool


### Starting the Application
1. The csv file should have columns: qid,question,image,solution,answer
1. Navigate to the project directory, e.g. matsci-bench/
2. Run the application:
```
python editing/app.py --csv path/to/your/dataset.csv
```

3. Open your browser and go to `http://[server_ip]:[port_number]` (for example, `http://scai1.cs.ucla.edu:5000`)

### Interface Overview

The interface consists of:
- **Left sidebar**: List of all available question IDs
- **Main content area**: Question editor with the following sections:
  - Question source and preview
  - Solution source and preview
  - Answer source and preview
  - Unit
  - Number of answers
  - Type
  - Notes
  - Image upload and display area
  - Navigation buttons

### Working with Questions

#### Browsing and Selecting Questions
1. The sidebar displays all available question IDs
2. Click on any ID to load that question
3. The currently selected question is highlighted

#### Editing Content
1. Edit text in the source textareas:
   - **Question Source**: The question text
   - **Solution Source**: The detailed solution
   - **Answer Source**: The short answer
   - **Unit**: The unit of the answer
   - **Number of answers**: The number of answers, single or multiple
   - **Type**: The type of the question, NUM, FORMULA, MCQ
   - **Notes**: Any additional notes
2. Click the **Save & Recompile** button to save chanegs to the csv and update the preview.
3. Uplaod images if the images are missing or wrong, images and the path will be saved right away.


#### Navigation
- Click **Previous** to go to the previous question in the list
- Click **Next** to go to the next question in the list
- Or choose a question id from the left sidebar.

## Example Questions

**Question:** 
A two-dimensional square body initially 1.00 cm by 1.00 cm was deformed into a rectangle 0.95 cm by 1.10 cm , as shown in the figure.
A. Calculate the strain, $e'_x$, along the diagonal from its initial and final dimensions. Then calculate the strains, $e_x$ and $e_y$, along the edges and use the transformation equation, $e_{ij}=\sum_m \sum_n \ell_{im} \ell_{jn} e_{mn}$, to find the strain along the diagonal. What is the absolute difference between the two values of $e'_{x}$.
B. Repeat A for a $1.00\,\text{cm}$ by $1.00\,\text{cm}$ square deformed into a $0.50\,\text{cm}$ by $2.0\,\text{cm}$ rectangle.
Give your answer as an ordered tuple (A's difference, B's difference).

**Solution:** 
A. The initial diagonal $= \sqrt{2} = 1.414214$, and for the small deformation, the final diagonal becomes $\sqrt{(0.95)^2 + (1.1)^2} = 1.4534$, so $e_{x'} = (L - L_{\mathrm{o}})/L_{\mathrm{o}} = L/L_{\mathrm{o}} - 1 = 1.4534/1.414214 - 1 = 0.0277$. Taking the angle, $\theta$, between the $x'$ and x (or y) axes as 45 degrees, $\varepsilon_{x'} = \ell_{x'x}^2 \varepsilon_x + \ell_{x'y}^2 \varepsilon_y = (1/2)(0.1) + (1/2)(-0.05) = 0.025$. The difference is $0.0277 - 0.025 = 0.0027$.

B. For the large deformation, the diagonal becomes $\sqrt{2^2 + 0.5^2} = 2.062$, so calculating the strain from this, $e_{x'} = 2.062/1.414214 - 1 = 0.4577$.
The strains on the edges are  $e_x = 1$ and $e_y = -0.5$, so $e'_x = \ell_{x'x}^2 e_x + \ell_{x'y}^2 e_y = \frac{1}{2}(1) + \frac{1}{2}(-0.5) = 0.25$. Therefore the difference is $0.4577 - 0.25 = 0.2077$.

**Answer:**
(0.0027, 0.2077)

**Unit:**

**Number of answers:** Multiple

**Type:** NUM

**Notes:**

**Image:** 
# MatSciBench

MatSciBench is a college-level materials science reasoning benchmark for evaluating large language models on quantitative, symbolic, and multimodal question answering.

- Paper: [MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Science](https://arxiv.org/abs/2510.12171)
- Dataset: [JunkaiZ/MatSciBench on Hugging Face](https://huggingface.co/datasets/JunkaiZ/MatSciBench)

The benchmark contains 1,340 materials science problems with reference solutions, final answers, units, difficulty labels, domain annotations, and embedded image inputs where needed. The public dataset is hosted on Hugging Face; this repository contains the evaluation code and prompting methods.

## Highlights

- Covers core materials science topics across materials, properties, structures, fundamental mechanisms, processes, and failure mechanisms.
- Includes numerical, formula, and multiple-choice answer formats.
- Provides step-by-step reference solutions for judging and error analysis.
- Supports text-only and multimodal model evaluation.
- Includes rule-based judging for numerical answers and optional LLM judging for formula answers.
- Implements multiple reasoning strategies: base prompting, tool augmentation, self-correction, and self-consistency.

## Repository Layout

```text
evaluation/   Evaluation entry point, judges, graders, and model registry
methods/      Prompting and reasoning methods used by the evaluator
utils/        API, dataset loading, image handling, and execution utilities
```

## Installation

Create the conda environment from the provided specification:

```bash
conda env create -f environment.yml
conda activate matsci-bench
```

The evaluator calls OpenAI-compatible chat completion endpoints. Set the API key for the model provider you plan to use:

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export DEEPSEEK_API_KEY=...
export QWEN_API_KEY=...
```

Only the key for the selected model is required. Model names, endpoints, and required environment variables are configured in `evaluation/model_registry.py`.

## Dataset

The default evaluation dataset is `JunkaiZ/MatSciBench`, split `test`.

```python
from datasets import load_dataset

dataset = load_dataset("JunkaiZ/MatSciBench", split="test")
print(dataset[0])
```

Important columns include:

- `qid`: question identifier
- `type`: answer type, `NUM` or `FORMULA`
- `question`: problem statement
- `image`: embedded image inputs, or an empty list
- `solution`: reference solution
- `answer`: final reference answer
- `unit`: expected unit
- `number_of_answers`: `single` or `multiple`
- `difficulty_level`: difficulty label
- `primary_category` and topic columns: domain annotations
- `source` and `original_qid`: source metadata

## Running Evaluation

Run the full benchmark with optional LLM judging for formula answers:

```bash
python evaluation/eval.py \
  --model gemini-2.5-flash \
  --method base \
  --llm_judge \
  --num_workers 8 \
  --output_dir results/evaluation
```

Supported methods:

- `base`: direct chain-of-thought prompting
- `tool`: prompts the model to write Python code, executes it, and asks for a final answer
- `correction`: self-correction after an initial answer
- `consistency`: self-consistency prompting with higher sampling temperature

The evaluator writes timestamped CSV files containing model outputs, extracted final answers, reference answers, judge decisions, judge reasoning, token counts, and image metadata.

## Model Registry

Add or edit model configurations in `evaluation/model_registry.py`.

Each model entry defines:

- the model name sent to the API
- the OpenAI-compatible endpoint URL
- the environment variable containing its API key
- whether the model supports multimodal inputs

For text-only models, the evaluator automatically skips questions with images. The `tool` and `correction` methods are currently restricted to text-only questions.

## Judging

The evaluation pipeline always applies rule-based judging where possible. Numerical answers are normalized and compared with symbolic and numeric tolerance logic. When `--llm_judge` is enabled, formula questions are additionally judged by the configured formula judge model in `evaluation/model_registry.py`.

The final CSV includes separate rule and LLM judge fields when both are available.

## Citation

If you use MatSciBench, please cite:

```bibtex
@misc{zhang2025matscibenchbenchmarkingreasoningability,
      title={MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Science},
      author={Junkai Zhang and Jingru Gan and Xiaoxuan Wang and Zian Jia and Changquan Gu and Jianpeng Chen and Yanqiao Zhu and Mingyu Derek Ma and Dawei Zhou and Ling Li and Wei Wang},
      year={2025},
      eprint={2510.12171},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12171},
}
```

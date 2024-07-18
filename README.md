The benchmark is available at [https://huggingface.co/datasets/ofir408/MedConceptsQA](https://huggingface.co/datasets/ofir408/MedConceptsQA).

If you use MedConceptsQA or find this repository useful for your research or work, please cite us using the following citation:
```
@article{shoham2024medconceptsqa,
  title={MedConceptsQA--Open Source Medical Concepts QA Benchmark},
  author={Shoham, Ofir Ben and Rappoport, Nadav},
  journal={arXiv preprint arXiv:2405.07348},
  year={2024}
}
```


**How To Run?**

Install the required dependencies:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Run the benchmark evaluation:
```
lm_eval --model hf --model_args pretrained=MODEL_ID --tasks med_concepts_qa --device cuda:0 --num_fewshot SHOTS_NUM --batch_size auto --limit 250 --output_path OUTPUT_RESULTS_DIR_PATH
```
Replace `MODEL_ID` with the model name (HuggingFace) or local path to the pretrained model you want to evaluate.

Replace `OUTPUT_RESULTS_DIR_PATH` with the directory path to store the results CSV file.

Replace `SHOTS_NUM` with the number of shots. The default is 4. For zero-shot learning, use 0. 

**Few-shot evaluation**: 
```
lm_eval --model hf --model_args pretrained=BioMistral/BioMistral-7B-DARE --tasks med_concepts_qa --device cuda:0 --num_fewshot 4 --batch_size auto --limit 250 --output_path  results/few_shot/250_examples/
```

**Zero-shot evaluation**:
 
```
lm_eval --model hf --model_args pretrained=BioMistral/BioMistral-7B-DARE --tasks med_concepts_qa --device cuda:0 --num_fewshot 0 --batch_size auto --limit 250 --output_path  results/few_shot/250_examples/
```


**Run GPT benchmark evaluation**:

Install the requirements:
`pip install -r requirements.txt`

SET `OPENAI_API_KEY` as an environment variable with your OpenAI key and then run with:
```
python gpt4_runner.py --model_id gpt-4-0125-preview --shots_num 4 --total_eval_examples_num 250 --output_results_dir_path results/few_shot/250_examples/
```
gpt-4 zero shot evaluation:
```
python gpt4_runner.py --model_id gpt-4-0125-preview --shots_num 0 --total_eval_examples_num 250 --output_results_dir_path results/zero_shot/250_examples/
```


## Leaderboard:

*Zero-shot*:

| Model Name                                | Accuracy | CI    |
|-------------------------------------------|----------|-------|
| gpt-4-0125-preview                        |**52.489**   | 3.573 |
| meta-llama/Meta-Llama-3-70B-Instruct      | 47.076   | 3.572 |
| aaditya/Llama3-OpenBioLLM-70B             | 41.849   | 3.53  |
| gpt-3.5-turbo                             | 37.058   | 3.456 |
| meta-llama/Meta-Llama-3-8B-Instruct       | 34.8     | 3.409 |
| aaditya/Llama3-OpenBioLLM-8B              | 29.431   | 3.262 |
| johnsnowlabs/JSL-MedMNX-7B                | 28.649   | 3.235 |
| epfl-llm/meditron-70b                     | 28.133   | 3.218 |
| dmis-lab/meerkat-7b-v1.0                  | 27.982   | 3.213 |
| BioMistral/BioMistral-7B-DARE             | 26.836   | 3.171 |
| epfl-llm/meditron-7b                      | 26.107   | 3.143 |
| dmis-lab/biobert-v1.1                     | 25.636   | 3.125 |
| UFNLP/gatortron-large                     | 25.298   | 3.111 |
| PharMolix/BioMedGPT-LM-7B                 | 24.924   | 3.095 |



*Few-shot*:

| Model Name                                | Accuracy | CI    |
|-------------------------------------------|----------|-------|
| gpt-4-0125-preview                        | **61.911**   | 3.475 |
| meta-llama/Meta-Llama-3-70B-Instruct      | 57.867   | 3.534 |
| aaditya/Llama3-OpenBioLLM-70B             | 53.387   | 3.57  |
| gpt-3.5-turbo                             | 41.476   | 3.526 |
| meta-llama/Meta-Llama-3-8B-Instruct       | 40.693   | 3.516 |
| aaditya/Llama3-OpenBioLLM-8B              | 35.316   | 3.421 |
| epfl-llm/meditron-70b                     | 34.809   | 3.409 |
| johnsnowlabs/JSL-MedMNX-7B                | 32.436   | 3.35  |
| BioMistral/BioMistral-7B-DARE             | 28.702   | 3.237 |
| PharMolix/BioMedGPT-LM-7B                 | 28.204   | 3.22  |
| dmis-lab/meerkat-7b-v1.0                  | 28.187   | 3.219 |
| epfl-llm/meditron-7b                      | 26.231   | 3.148 |
| dmis-lab/biobert-v1.1                     | 25.982   | 3.138 |
| UFNLP/gatortron-large                     | 25.093   | 3.102 |



If you wish to submit your model for evaluation, please open us a GitHub issue with your model's HuggingFace name.

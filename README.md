The benchmark is available at [https://huggingface.co/datasets/ofir408/MedConceptsQA](https://huggingface.co/datasets/ofir408/MedConceptsQA).

**How To Run?**

Install the required dependencies:
```
pip install -r requirements.txt
```

Run the benchmark evaluation:
```
python evaluation_runner.py --model_id MODEL_ID --output_results_dir_path OUTPUT_RESULTS_DIR_PATH --shots_num SHOTS_NUM --total_num_experiments NUM_EXPERIMENTS
```
Replace `MODEL_ID` with the model name (HuggingFace) or local path to the pretrained model you want to evaluate.

Replace `OUTPUT_RESULTS_DIR_PATH` with the directory path to store the results CSV file.

Replace `SHOTS_NUM` with the number of shots. The default is 4. For zero-shot learning, use 0. 

Replace `NUM_EXPERIMENTS` with the number of experiments to use for calculating the 95% condifence intervals. The default is 1.


**Few-shot evaluation**: 

`python evaluation_runner.py --model_id BioMistral/BioMistral-7B-DARE --total_eval_examples_num 250 --output_results_dir_path results/few_shot/250_examples/ --shots_num 4`

**Zero-shot evaluation**:
 
`python evaluation_runner.py --model_id BioMistral/BioMistral-7B-DARE --total_eval_examples_num 250 --output_results_dir_path results/zero_shot/250_examples/ --shots_num 0`


**Run GPT benchmark evaluation**:

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

| model                                   | mean_accuracy | ci   |
|-----------------------------------------|---------------|------|
| gpt-4-0125-preview                      | **52.489**        | 3.135|
| gpt-3.5-turbo                           | 37.058        | 2.399|
| dmis-lab/biobert-v1.1                   | 26.151        | 3.571|
| meta-llama/Meta-Llama-3-8B-Instruct     | 25.840        | 6.199|
| epfl-llm/meditron-7b                    | 25.751        | 3.340|
| dmis-lab/meerkat-7b-v1.0                | 25.680        | 3.983|
| HuggingFaceH4/zephyr-7b-beta            | 25.538        | 3.075|
| epfl-llm/meditron-70b                   | 25.360        | 2.630|
| yikuan8/Clinical-Longformer             | 25.040        | 2.406|
| UFNLP/gatortron-medium                  | 24.862        | 3.170|
| PharMolix/BioMedGPT-LM-7B               | 24.747        | 4.219|
| BioMistral/BioMistral-7B-DARE           | 24.569        | 3.867|
| johnsnowlabs/JSL-MedMNX-7B              | 24.427        | 3.185|

*Few-shot*:

| model                                   | mean_accuracy | ci   |
|-----------------------------------------|---------------|------|
| gpt-4-0125-preview                      | **61.911**        | 2.320|
| gpt-3.5-turbo                           | 41.476        | 2.481|
| meta-llama/Meta-Llama-3-8B-Instruct     | 25.653        | 2.707|
| johnsnowlabs/JSL-MedMNX-7B              | 25.627        | 2.497|
| yikuan8/Clinical-Longformer             | 25.547        | 3.495|
| dmis-lab/biobert-v1.1                   | 25.458        | 2.649|
| epfl-llm/meditron-70b                   | 25.262        | 3.499|
| BioMistral/BioMistral-7B-DARE           | 25.058        | 2.676|
| HuggingFaceH4/zephyr-7b-beta            | 25.058        | 2.121|
| dmis-lab/meerkat-7b-v1.0                | 24.942        | 2.879|
| PharMolix/BioMedGPT-LM-7B               | 24.924        | 3.363|
| epfl-llm/meditron-7b                    | 23.787        | 3.496|


If you wish to submit your model for evaluation, please open us a GitHub issue with your model's HuggingFace name.

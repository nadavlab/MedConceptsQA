import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Length of IterableDataset.*")
import torch
from tqdm import tqdm

HF_CACHE_DIR = '/sise/nadav-group/nadavrap-group/ofir/hf_cache'  # can be None if you don't want to use custom cache dir.
if HF_CACHE_DIR:
    os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
    os.environ['HF_HOME'] = HF_CACHE_DIR

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoTokenizer


def zero_shot_classification(pipeline, tokenizer, questions, candidate_labels, batch_size, vocab_name, level):
    predictions = []
    for idx in tqdm(range(0, len(questions), batch_size), desc=f'Inference {vocab_name} vocab, {level} level'):
        batch_questions = questions[idx: idx + batch_size]
        batch_questions_with_template = [to_instruct_template(question, tokenizer) for question in batch_questions]
        batch_predictions = pipeline(batch_questions_with_template, candidate_labels)
        predictions.extend(batch_predictions)
    return predictions


def to_instruct_template(text, tokenizer):
    messages = [
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def process_vocabulary(data, tokenizer, question_column, answer_id_column, zero_shot_pipeline, batch_size):
    data = data.to_pandas()
    vocabularies = data['vocab'].unique()
    levels = data['level'].unique()
    results = []
    for vocab in vocabularies:
        for level in levels:
            query = f"vocab=='{vocab}' & level=='{level}'"
            vocab_level_data = data.query(query)
            total_examples = vocab_level_data.shape[0]
            train_data, test_data = train_test_split(vocab_level_data, test_size=min(total_examples, 1_000))

            sampled_questions = test_data[question_column].tolist()
            answer_ids = test_data[answer_id_column].tolist()
            predictions = zero_shot_classification(zero_shot_pipeline, tokenizer, sampled_questions, [1, 2, 3, 4],
                                                   batch_size, vocab_name=vocab, level=level)
            accuracy = accuracy_score(answer_ids, [pred['labels'][0] for pred in predictions])
            report = classification_report(answer_ids, [pred['labels'][0] for pred in predictions], output_dict=True)

            result = {
                'Model': zero_shot_pipeline.model.config.name_or_path,
                'Level': level,
                'Vocabulary': vocab,
                'Accuracy': accuracy,
                'Num_Samples': len(sampled_questions),
                'Classification_Report': report
            }
            results.append(result)

    return results


def main(model_id, dataset_name, output_results_dir_path):
    print('Loading the dataset..')
    dataset = load_dataset(dataset_name, cache_dir=HF_CACHE_DIR)
    print(f'Done to load the dataset. Dataset={dataset}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    zero_shot_pipeline = pipeline("zero-shot-classification", model=model_id,
                                  device_map='auto', torch_dtype=torch.bfloat16)
    results = process_vocabulary(dataset['train'], tokenizer, 'question', 'answer_id', zero_shot_pipeline, batch_size=1)

    df = pd.DataFrame(results)
    print(f"results={df}")
    os.makedirs(f"{output_results_dir_path}/{model_id}", exist_ok=True)
    print(f'writing results to dir_path={output_results_dir_path}')
    results_csv_path = f"{output_results_dir_path}/{model_id}/results.csv" if output_results_dir_path is not None else "results.csv"
    df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for zero-shot classification")
    parser.add_argument("--model_id", type=str, help="The model name to evaluate", required=True)

    parser.add_argument("--dataset_name", type=str, default="ofir408/try1",
                        help="Name of the dataset to load using load_dataset", required=False)
    parser.add_argument("--output_results_dir_path", type=str, help="Path to store the results CSV files",
                        default="results.csv", required=False)
    args = parser.parse_args()

    main(args.model_id, args.dataset_name, args.output_results_dir_path)

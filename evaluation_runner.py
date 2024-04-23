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
from transformers import pipeline, AutoTokenizer
import random

def zero_shot_classification(pipeline, tokenizer, questions, candidate_labels, batch_size, vocab_name, level):
    predictions = []
    for idx in tqdm(range(0, len(questions), batch_size), desc=f'Inference {vocab_name} vocab, {level} level'):
        batch_questions = questions[idx: idx + batch_size]
        batch_questions_with_template = [to_instruct_template(question, tokenizer) for question in batch_questions]
        batch_predictions = pipeline(batch_questions_with_template, candidate_labels, max_new_tokens=1, temperature=0)
        predictions.extend(batch_predictions)
    return predictions


def to_instruct_template(text, tokenizer):
    messages = [
        # {"role": "system", "content": "Answer the multiple-choice question about medical knowledge.\n\n"},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def process_vocabulary(data: pd.DataFrame, few_shot_data: pd.DataFrame, tokenizer, question_column, answer_id_column, zero_shot_pipeline, batch_size,
                                 shots_num, total_eval_examples_num):
    vocabularies = data['vocab'].unique()
    levels = data['level'].unique()
    results = []
    for vocab in vocabularies:
        for level in levels:
            query = f"vocab=='{vocab}' & level=='{level}'"
            few_shot_vocab_level_data = few_shot_data.query(query)
            vocab_level_data = data.query(query)
            few_shot_examples_prompt = create_few_shot_example(df=few_shot_vocab_level_data, shots_num=shots_num)
            total_examples = vocab_level_data.shape[0]
            test_data = vocab_level_data.sample(n=min(total_examples, total_eval_examples_num))

            sampled_questions = test_data[question_column].tolist()
            prefix = "Answer A,B,C,D according to the answer to this multiple choice question.\n"
            suffix = "\nAnswer:"
            sampled_questions_full_prompt = [
                prefix + few_shot_examples_prompt +
                ("\n" if len(few_shot_examples_prompt) > 0 else "") + question + suffix for question
                in sampled_questions]
            predictions = zero_shot_classification(zero_shot_pipeline, tokenizer, sampled_questions_full_prompt,
                                                   ['A', 'B', 'C', 'D'],
                                                   batch_size, vocab_name=vocab, level=level)
            answer_ids = test_data[answer_id_column].tolist()
            accuracy = accuracy_score(answer_ids, [pred['labels'][0] for pred in predictions])
            print(f"vocab={vocab}, level={level}, accuracy={accuracy}")
            report = classification_report(answer_ids, [pred['labels'][0] for pred in predictions], output_dict=True)

            result = {
                'Model': zero_shot_pipeline.model.config.name_or_path,
                'Level': level,
                'Vocabulary': vocab,
                'Accuracy': accuracy,
                'Num_Samples': len(sampled_questions),
                'Classification_Report': report,
                'Shots_num': shots_num,
                'Answers': answer_ids,
                'Predictions': predictions,
                'Sampled_questions': sampled_questions_full_prompt
            }
            results.append(result)

    return results


def create_few_shot_example(df: pd.DataFrame, shots_num: int) -> str:
    if shots_num == 0:
        return ""  # zero shot learning.

    shot_examples = df.head(shots_num)
    final_shot_prompt = ""
    for _, example in shot_examples.iterrows():
        question = example["question"]
        answer_id = example["answer_id"]
        example_prompt = f"{question}\nAnswer:{answer_id}\n\n".replace("  ", "")
        final_shot_prompt += example_prompt
    return final_shot_prompt


def main(model_id, dataset_name, output_results_dir_path, shots_num, total_eval_examples_num):
    print('Loading the dataset..')
    dataset = load_dataset(dataset_name, cache_dir=HF_CACHE_DIR)
    print(f'Done to load the dataset. Dataset={dataset}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    zero_shot_pipeline = pipeline("zero-shot-classification", model=model_id,
                                  device='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
    results = process_vocabulary(data=dataset['train'].to_pandas(), few_shot_data=dataset["dev"].to_pandas(), tokenizer=tokenizer,
                                 question_column='question', answer_id_column='answer_id', zero_shot_pipeline=zero_shot_pipeline, batch_size=1,
                                 shots_num=shots_num, total_eval_examples_num=total_eval_examples_num)

    df = pd.DataFrame(results)
    print(f"results={df}")
    os.makedirs(f"{output_results_dir_path}/{model_id}", exist_ok=True)
    print(f'writing results to dir_path={output_results_dir_path}')
    rand_num = random.randint(1, 10000000)

    results_csv_path = f"{output_results_dir_path}/{model_id}/results_{rand_num}.csv" if output_results_dir_path is not None else f"results_{rand_num}.csv"
    df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for zero-shot classification")
    parser.add_argument("--model_id", type=str, help="The model name to evaluate", required=True)
    parser.add_argument("--dataset_name", type=str, default="ofir408/try1_v2",
                        help="Name of the dataset to load using load_dataset", required=False)
    parser.add_argument("--output_results_dir_path", type=str, help="Directory path to store the results CSV files",
                        default="results", required=False)
    parser.add_argument("--shots_num", type=int, help="Number of few shot examples",
                        default=4, required=False)
    parser.add_argument("--total_eval_examples_num", type=int,
                        help="Number of examples for evaluation per dataset",
                        default=250, required=False)

    args = parser.parse_args()

    main(args.model_id, args.dataset_name, args.output_results_dir_path, args.shots_num, args.total_eval_examples_num)

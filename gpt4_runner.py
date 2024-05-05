import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore", message="Length of IterableDataset.*")

HF_CACHE_DIR = '/sise/nadav-group/nadavrap-group/ofir/hf_cache'  # can be None if you don't want to use custom cache dir.
if HF_CACHE_DIR:
    os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
    os.environ['HF_HOME'] = HF_CACHE_DIR

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

import openai
from tqdm import tqdm

tqdm.pandas()

client = openai.OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))


def call_openai(model_id, prompt: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_id,
    )
    print(response)
    gpt_final_response = response.choices[0].message.content
    print(f"response={gpt_final_response}")
    return gpt_final_response


def parse_gpt_response(gpt_response: str) -> str:
    return gpt_response.split(".")[0].replace(" ", "")


def zero_shot_classification(model_id, questions, vocab_name, level):
    predictions = []
    for question in tqdm(questions, desc=f'Inference {vocab_name} vocab, {level} level'):
        gpt_response = call_openai(model_id=model_id, prompt=question)
        try:
            final_prediction = parse_gpt_response(gpt_response)
            predictions.append(final_prediction)
        except Exception as e:
            print(f"failed to parse gpt_response={gpt_response}, exception={e}")
    return predictions


def process_vocabulary(model_id, data: pd.DataFrame, few_shot_data: pd.DataFrame, question_column, answer_id_column,
                       shots_num: int,
                       total_eval_examples_num: int):
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
            predictions = zero_shot_classification(model_id, sampled_questions_full_prompt, vocab, level)
            answer_ids = test_data[answer_id_column].tolist()
            accuracy = accuracy_score(answer_ids, predictions)
            print(f"vocab={vocab}, level={level}, accuracy={accuracy}")
            report = classification_report(answer_ids, predictions, output_dict=True)

            result = {
                'Model': model_id,
                'Vocabulary': vocab,
                'Level': level,
                'Accuracy': accuracy,
                'Num_Samples': len(sampled_questions),
                'Shots_num': shots_num,
                'Classification_Report': report,
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
    results = process_vocabulary(model_id, dataset['train'].to_pandas(), dataset["dev"].to_pandas(),
                                 'question', 'answer_id',
                                 shots_num=shots_num, total_eval_examples_num=total_eval_examples_num)

    df = pd.DataFrame(results)
    sorting_order = ['easy', 'medium', 'hard']
    df['Level'] = pd.Categorical(df['Level'], categories=sorting_order, ordered=True)
    df = df.sort_values(by=['Vocabulary', 'Level'])

    print(f"results={df}")
    os.makedirs(f"{output_results_dir_path}/{model_id}", exist_ok=True)
    print(f'writing results to dir_path={output_results_dir_path}')
    rand_num = random.randint(1, 10000000)
    results_csv_path = f"{output_results_dir_path}/{model_id}/results_{rand_num}.csv" if output_results_dir_path is not None else f"results_{rand_num}.csv"
    df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for zero-shot classification")
    parser.add_argument("--model_id", type=str, help="The model name to evaluate", required=True)
    parser.add_argument("--dataset_name", type=str, default="ofir408/MedConceptsQA",
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

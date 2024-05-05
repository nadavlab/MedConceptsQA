import pyhealth
import random
import pandas as pd
from pydantic import BaseModel, validator
from typing import Dict, List
from pyhealth.medcode import InnerMap
from tqdm import tqdm
import multiprocessing
from functools import partial
import networkx as nx

OUTPUT_DIR = "/sise/nadav-group/nadavrap-group/ofir/medical_qa_benchmark/v2"


class QuestionsDataset(BaseModel):
    vocab_name: str
    level: str
    answer_and_options: Dict[str, List[str]]

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['easy', 'medium', 'hard']
        if v.lower() not in valid_levels:
            raise ValueError("Invalid level. Must be one of: 'easy', 'medium', 'hard'.")
        return v.lower()


def is_leaf(vocab, node_name: str):
    return len(vocab.get_descendants(node_name)) == 0


def create_easy_questions(vocab_name: str, num_of_options: int, only_leafs: bool):
    question_and_options = dict()
    vocab = InnerMap.load(vocab_name)
    nodes = list(vocab.graph.nodes)
    if only_leafs:
        nodes = list(filter(lambda node_name: is_leaf(vocab, node_name), nodes))

    for node_name in tqdm(nodes):
        leaf_nodes_options = list(set(nodes) - set(node_name))
        random_codes = random.sample(leaf_nodes_options, num_of_options)
        question_and_options[node_name] = random_codes
    return QuestionsDataset(vocab_name=vocab_name,
                            level='easy',
                            answer_and_options=question_and_options)


def create_questions_by_distance(vocab_name: str, level: str, required_edges_distance: List[int], num_of_options: int):
    question_and_options = dict()
    vocab = InnerMap.load(vocab_name)
    nodes = vocab.graph.nodes
    undirected_vocab_graph = vocab.graph.to_undirected()

    for node_name in tqdm(nodes):
        shortest_paths_for_node = nx.single_source_shortest_path_length(undirected_vocab_graph, node_name,
                                                                        cutoff=max(required_edges_distance))
        filtered_shortest_paths_for_node = {node_name: distance for node_name, distance
                                            in shortest_paths_for_node.items()
                                            if distance in required_edges_distance}
        possible_nodes = filtered_shortest_paths_for_node.keys()
        if len(possible_nodes) >= num_of_options:  # TODO: CHANGE THE LOGIC IF WE WANT!!
            random_codes = random.sample(possible_nodes, num_of_options)
            question_and_options[node_name] = random_codes
    return QuestionsDataset(vocab_name=vocab_name,
                            level=level,
                            answer_and_options=question_and_options)


def create_hard_questions(vocab_name: str, num_of_options: int):
    return create_questions_by_distance(vocab_name, 'hard', [2], num_of_options)


def create_medium_questions(vocab_name: str, num_of_options: int):
    return create_questions_by_distance(vocab_name, 'medium', [3, 4, 5], num_of_options)


def generate_question(vocab, medical_code: str, option1: str, option2: str, option3: str, option4: str) -> str:
    vocab_name = vocab.vocabulary
    return f"""
    What is the description of the medical code {medical_code} in {vocab_name}?
    A. {vocab.lookup(option1)}
    B. {vocab.lookup(option2)}
    C. {vocab.lookup(option3)}
    D. {vocab.lookup(option4)}
    """


def write_dataset(output_dir: str, vocab, dataset: QuestionsDataset):
    vocab_name = dataset.vocab_name
    level = dataset.level
    output_path = f"{output_dir}/{vocab_name}_{level}.csv"
    answer_and_options = dataset.answer_and_options
    shuffled_data = []
    for key, values in answer_and_options.items():
        try:
            options = list(values) + [key]
            options = random.sample(options, len(options))
            question = generate_question(vocab=vocab, medical_code=key, option1=options[0], option2=options[1],
                                         option3=options[2], option4=options[3])
            answer_id = options.index(key) + 1
            answer_id_mapping = {1: 'A', 2: 'B', 4: 'C', 4: 'D'}
            answer_id = answer_id_mapping[answer_id]

            row = {'answer': vocab.lookup(key), 'answer_id': answer_id, 'option1': vocab.lookup(options[0]),
                   'option2': vocab.lookup(options[1]), 'option3': vocab.lookup(options[2]),
                   'option4': vocab.lookup(options[3]), 'question': question}
            shuffled_data.append(row)
        except Exception as e:
            print(f"exception in write_dataset, possible in generate_question.. Exception= {e}")

    shuffled_dataset = pd.DataFrame(shuffled_data)
    shuffled_dataset['vocab'] = vocab_name
    shuffled_dataset['level'] = level
    shuffled_dataset = shuffled_dataset.sample(frac=1).reset_index(drop=True)
    shuffled_dataset.index += 1
    shuffled_dataset.to_csv(output_path, index=True, index_label='question_id')
    print(f'done processing vocab_name={vocab_name}, level={level}')


def process_vocab(vocab_name, num_of_options, output_dir):
    vocab = InnerMap.load(vocab_name)
    easy_dataset = create_easy_questions(vocab_name=vocab_name, num_of_options=num_of_options, only_leafs=False)
    write_dataset(output_dir=output_dir, vocab=vocab, dataset=easy_dataset)
    medium_dataset = create_medium_questions(vocab_name=vocab_name, num_of_options=num_of_options)
    write_dataset(output_dir=output_dir, vocab=vocab, dataset=medium_dataset)
    hard_dataset = create_hard_questions(vocab_name=vocab_name, num_of_options=num_of_options)
    write_dataset(output_dir=output_dir, vocab=vocab, dataset=hard_dataset)


vocabularies = ["ICD10PROC", "ICD9PROC", "ICD9CM", "ICD10CM", "ATC"]

for v in vocabularies:
    process_vocab(v, 3, OUTPUT_DIR)

partial_process_vocab = partial(process_vocab, num_of_options=3, output_dir=OUTPUT_DIR)
with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    pool.map(partial_process_vocab, vocabularies)

print('done!')

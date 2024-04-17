import os
import re
import time
from typing import List

import numpy as np
import together


def generate_together_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together API.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """

    client = together.Together()
    outputs = client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]


def main():
    # get filepaths from directory and load texts
    filepaths = get_filepaths(DATA_DIR)
    list_text = []
    list_category_name = []
    nodes = []
    for f_path in filepaths:
        text = read_text_file(f_path)
        name = os.path.basename(f_path)
        category_name = re.sub(r"\d+_([^.]*)", r"\1", name)
        list_category_name.append(category_name)
        name = re.sub(r"\.txt", r"", name)
        node = {"id": name, "cluster_id": None, "embedding": None}
        nodes.append(node)
        list_text.append(text)
    t0 = time.time()

    # Define values.
    model_names = [
        "togethercomputer/m2-bert-80M-32k-retrieval",
        "togethercomputer/m2-bert-80M-8k-retrieval",
        "togethercomputer/m2-bert-80M-2k-retrieval",
        "sentence-transformers/msmarco-bert-base-dot-v5",
        # "WhereIsAI/UAE-Large-V1", # produces error
        # "BAAI/bge-large-en-v1.5", # produces error
        # "bert-base-uncased", # produces error
    ]
    # context_length = 8192
    # num_samples = 100
    for model_name in model_names:
        embeddings = generate_together_embeddings(list_text, model_name)
        print(f"{model_name} - Time taken: {time.time() - t0:.2f} seconds")
        embeddings_for_clustering = [np.array([embedding]) for embedding in embeddings]
        for node, embedding in zip(nodes, embeddings):
            node["embedding"] = np.array([embedding])
        # Cluster the embeddings
        model_name = re.sub(r"/", r"_", model_name)
        save_path = f"/Users/eric/Library/CloudStorage/Dropbox/git/github/articulo/conversation_clustering/data/plots/{model_name}_sample_v4.png"


# def sample_and_extract_embeddings(
#     data_path: str, model_api_string: str, num_samples: int = -1, context_length=512
# ) -> np.ndarray:
#     """Sample data examples and extract embeddings for each example.

#     Args:
#         data_path: str. A path to the data file. It should be in the .jsonl format.
#         model_api_string: str. An API string for a specific embedding model of your choice.
#         num_samples: int. The number of data examples to sample.
#         context_length: int. The max context length of the model (model_api_string).

#     Returns:
#         embeddings: np.ndarray with num_samples by m where m is the embedding dimension.

#     """
#     max_num_chars = context_length * 4  # Assuming that each token is about 4 characters.
#     embeddings = []
#     count = 0
#     print(f"Reading from {data_path}")
#     with open(data_path, "r") as f:
#         for line in f:
#             try:
#                 # ex = json.loads(line)
#                 # read filepath
#                 total_chars = len(ex["text"])
#                 if total_chars < max_num_chars:
#                     text_ls = [ex["text"]]
#                 else:
#                     text_ls = [ex["text"][i : i + max_num_chars] for i in range(0, total_chars, max_num_chars)]
#                 embeddings.extend(
#                     generate_together_embeddings(text_ls[: min(len(text_ls), num_samples - count)], model_api_string)
#                 )
#                 count += min(len(text_ls), num_samples - count)
#             except Exception as e:
#                 print(f"Error occurred while loading the JSON file of {data_path} with the error message {e}.")
#             if count >= num_samples:
#                 break
#     return np.array(embeddings)

import string

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
from nltk.chunk import ne_chunk_sents
from nltk.tag import pos_tag_sents
import numpy as np
import nltk
from copy import copy
import os

nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('averaged_perceptron_tagger')


class ModelData(Dataset):
    def __init__(self, data_folder: str, embedding_ind: dict, subset: str = "train") -> None:
        """
        :param data_folder: Folder where train, test and validation data is present
        :param embedding_ind: A dictionary mapping each token to its index in the embedding matrix
        :param subset: Which split, train, test or dev/validation
        """
        self.tokens = dict()  # Dictionary containing tokenized sentence per index
        self.labels = dict()  # Dictionary containing labels per index
        self.index_emb = embedding_ind  # Dictionary mapping from string token to vocabulary index

        tags = ["SENTIMENT", "CHANGE", "ACTION", "SCENARIO", "POSSESSION"]  # Set of events

        self.target_encoder = LabelEncoder().fit(
            np.array(["O"] + ["B-" + x for x in tags] + ["I-" + x for x in tags]).T
        )  # Build encoder to decode/encode labels

        # Open file stream and start reading jsonl
        with open(os.path.join(data_folder, subset) + ".jsonl", "r") as file:
            for line in file:
                line = json.loads(line)
                self.tokens[line["idx"]] = line["tokens"]
                # ignore label for punctuation (it would be 0)
                self.labels[line["idx"]] = \
                    [y for x, y in zip(line['tokens'], line['labels']) if x not in string.punctuation]
                if not self.labels[line["idx"]]:  # If label list is empty, this is a sentence made up of punctuation
                    self.labels[line["idx"]] = ['O']  # Just add 0 label in order to allow digesting

        pos_tagged = pos_tag_sents(self.tokens.values())  # Perform POS tagging on sentences
        chunked = ne_chunk_sents(pos_tagged, binary=True)  # Chunk output of POS tagging for NE recognition
        self.chunked = dict(zip(self.tokens.keys(), chunked))  # Save output in a dictionary to speed up lookup

    def __getitem__(self, item):
        labels = self.target_encoder.transform(self.labels[item])
        tree = self.chunked[item]
        embeddings, rep_mask, pos_tags = sample_builder(tree, self.index_emb)

        grouped_labels = list()
        i = 0
        for group_len in rep_mask:
            grouped_labels.append(labels[i])
            i += group_len

        return (
            torch.tensor(embeddings, dtype=torch.int32),
            torch.tensor(grouped_labels, dtype=torch.int64),
            rep_mask,
            torch.tensor(pos_tags, dtype=torch.int32)
        )

    def __len__(self):
        return len(self.tokens)


def sample_builder(tree: nltk.Tree, index_emb: dict[str:int]) -> tuple[list[np.array], list[int, ...], list[int, ...]]:
    """
    The function takes care of building the actual embedding samples + a "rep mask" which allows tracking the
    length of multi-token expressions which are embedded by a single vector (since they are named entities). This
    extra step is useful downstream. Additionally, it also outputs integers encoding POS tagging of the tokens.
    It takes as input the NLTK tree outputted by the named entity chunking.
    :param tree: The Tree class from NLTK containing named entity (binary) tagging.
    :param index_emb: The dictionary mapping tokens/entities to vocabulary indexes
    :return: List of embeddings + list of the lengths of the expressions embedded + list of pos tag integer encoding
    """
    tree = copy(tree)  # Avoid side effect on obj by shallow copying

    embeddings_ind = list()  # Initialize list accumulating lists of vocabulary indexes
    rep_mask = list()  # Initialize list of lists containing the "rep mask" useful to track multi-token entities
    pos_tags = list()  # Initialize list of lists to store POS tag indexes

    while tree:  # Start NLTK tree parsing
        node = tree.pop(0)  # Pop starting node of the tree
        if isinstance(node, tuple):  # If the node is a single token
            if node[0] in string.punctuation:  # If it is punctuation, ignore and skip iteration
                continue
            rep_mask.append(1)  # Add 1 to rep mask to signal that the first embedding is related to a 1 word token
            try:
                embeddings_ind.append(index_emb[node[0].lower()])  # Add vocabulary index through dictionary
            except KeyError:  # If token not found, add out of vocabulary token
                embeddings_ind.append(len(index_emb))
            pos_tags.append(tag2int(node[1]))  # Add transformed to int POS tag

        elif isinstance(node, nltk.Tree):  # If the node is another tree, we are dealing with a named entity
            leaves = list(node)
            token_set = [x[0].lower() for x in leaves]
            key = "entity/" + "_".join(token_set)
            try:
                embeddings_ind.append(index_emb[key])  # Search for the entity in the dictionary
                # Possibly more than 1 word for a single embedding, let's keep this into consideration with rep_mask
                rep_mask.append(len([x for x in token_set if x not in string.punctuation]))
                pos_tags.append(2)  # A named entity is a name, add related pos tag
            except KeyError:  # If named entity not found,
                # revert to parsing the single tokens forming it as std words
                for i, token in enumerate(token_set):
                    if token in string.punctuation:  # Difficult to have an entity with punctuation, but just in case
                        continue
                    rep_mask.append(1)  # Add 1 to rep mask because the embedding is related to a single token
                    pos_tags.append(2)  # Add pos tag, each single token is considered as name (since part of a NE)
                    try:
                        embeddings_ind.append(index_emb[token.lower()])  # Search embedding for single token
                    except KeyError:
                        embeddings_ind.append(len(index_emb))  # Go OOV
    # It may happen that some sentences are made up only of punctuation --> empty list of indexes
    if not embeddings_ind:  # 'empty' sentence makes the model crash, consider it as a single OOV token sentence
        embeddings_ind.append(len(index_emb))
        rep_mask.append(1)
        pos_tags.append(5)
    return embeddings_ind, rep_mask, pos_tags


def obs_collate(batch: list[torch.tensor, torch.tensor, list[int], torch.tensor]):
    """
    Simple utility function for DataLoader module.
    """
    embeddings = [obs[0] for obs in batch]
    labels = [obs[1] for obs in batch]
    len_masks = [obs[2] for obs in batch]
    pos_tags = [obs[3] for obs in batch]
    embeddings = torch.nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False)
    labels = torch.nn.utils.rnn.pack_sequence(labels, enforce_sorted=False)
    pos_tags = torch.nn.utils.rnn.pack_sequence(pos_tags, enforce_sorted=False)
    return embeddings, labels, len_masks, pos_tags


def tag2int(tag: str) -> int:
    """
    Simple utility function to map from Penn Treebank POS Tags to an integer encoding.
    :param tag: The tag
    :return: The integer encoding of the tag
    """
    if tag.startswith('V'):
        return 1
    elif tag.startswith('N'):
        return 2
    elif tag.startswith('J'):
        return 3
    elif tag.startswith('RB'):
        return 4
    else:
        return 5




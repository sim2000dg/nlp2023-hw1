import math

import requests
import bz2
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import json
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from seqeval.metrics import f1_score
import itertools

nltk.download("maxent_ne_chunker")
nltk.download("words")


def download_embeddings(dimension: str, path: str) -> str:
    if os.path.exists(os.path.join(path, "embeddings.txt")):
        print("The embeddings are already downloaded.")
        return os.path.join(path, "embeddings.txt")

    if dimension == "small":
        url = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_100d.txt.bz2"
    elif dimension == "medium":
        url = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_300d.txt.bz2"
    elif dimension == "big":
        url = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_500d.txt.bz2"
    else:
        raise ValueError(
            "The dimension argument must be either 'small', 'medium' or 'big'"
        )

    r = requests.get(url, stream=True)
    total_size = int(r.headers["Content-Length"])

    with open(os.path.join(path, "embeddings.bz2"), "wb") as f:
        for chunk in tqdm(
            r.iter_content(10 ** 7), total=math.ceil(total_size / (10 ** 7))
        ):
            f.write(chunk)
    print("Embeddings downloaded, decompression started")

    with bz2.BZ2File(os.path.join(path, "embeddings.bz2")) as stream, open(
        os.path.join(path, "embeddings.txt"), "wb"
    ) as output:
        shutil.copyfileobj(stream, output, 10 ** 7)

    print("Decompression ended")

    os.remove(os.path.join(path, "embeddings.bz2"))

    return os.path.join(path, "embeddings.txt")


def load_embeddings(path: str):
    embeddings = pd.read_csv(
        path, sep=" ", header=None, index_col=0, skiprows=1
    ).astype(np.float32)
    return embeddings


class ModelData(Dataset):
    def __init__(
        self,
        data_folder: str,
        embeddings: pd.DataFrame,
        subset: str = "train",
    ) -> None:
        self.tokens = dict()
        self.labels = dict()
        self.embeddings = embeddings
        self.oov = (
            np.random.default_rng(1234)
            .normal(
                loc=self.embeddings.mean().mean(),
                scale=self.embeddings.std().mean(),
                size=len(self.embeddings.columns),
            )
            .astype(np.float32)
        )
        tags = ["SENTIMENT", "CHANGE", "ACTION", "SCENARIO", "POSSESSION"]

        self.target_encoder = LabelEncoder().fit(
            np.array(["O"] + ["B-" + x for x in tags] + ["I-" + x for x in tags]).T
        )

        with open(os.path.join(data_folder, subset) + ".jsonl", "r") as file:
            for line in file:
                line = json.loads(line)
                self.tokens[line["idx"]] = line["tokens"]
                self.labels[line["idx"]] = line["labels"]

    def sample_builder(self, tree: nltk.Tree) -> tuple[list[np.array], list[int]]:
        """
        The method takes care of building the actual embedding samples + a "length mask" which allows tracking the
        length of multi-token expressions which are embedded by a single vector (since they are named entities). This
        extra step is useful downstream.
        :param tree: The Tree class from NLTK containing named entity (binary) tagging.
        :return: List of embeddings + list of the lengths of the expressions embedded
        """
        embeddings = list()
        len_mask = list()
        while tree:
            node = tree.pop(0)
            if isinstance(node, tuple):
                len_mask.append(1)
                if node[1] == "NE":
                    key = "ENTITY/" + node[1]
                    try:
                        embeddings.append(self.embeddings.loc[key])
                        continue
                    except KeyError:
                        pass
                try:
                    embeddings.append(self.embeddings[node[1].lower()])
                except KeyError:
                    embeddings.append(self.oov)

            elif isinstance(node, nltk.Tree) and node.label() == "NE":
                token_set = [x[0] for x in list(node)]
                key = "ENTITY/" + "_".join([x for x in list(token_set)])
                try:
                    embeddings.append(self.embeddings.loc[key])
                    len_mask.append(len(node))
                except KeyError:
                    for token in token_set:
                        len_mask.append(1)
                        try:
                            embeddings.append(self.embeddings.loc[token.lower()])
                        except KeyError:
                            embeddings.append(self.oov)

        return embeddings, len_mask

    def __getitem__(self, item):
        strings = self.tokens[item]
        labels = self.target_encoder.transform(self.labels[item])
        grouped_labels = list()
        strings = pos_tag(strings)
        tree = ne_chunk(strings, binary=True)
        embeddings, len_mask = self.sample_builder(tree)
        i = 0
        while True:
            grouped_labels.append(labels[i])
            if i < len(labels):
                break
            i += len_mask[i]

        return embeddings, grouped_labels

    def __len__(self):
        return len(self.tokens)


def trainer(
    model_,
    epochs: int,
    learning_r: float,
    dataloaders: list[DataLoader, DataLoader],
    torch_device: torch.device,
):
    model_.to(torch_device)
    optimizer = torch.optim.Adam(
        model_.parameters(), lr=learning_r
    )  # Adam optimizer initialization
    loss_history = list()  # Init loss history
    val_loss = list()  # List of validation losses
    seq_F1 = list()
    for _ in (p_bar := tqdm(range(epochs), total=epochs, position=0, leave=True)):
        for stage in ["train", "valid"]:
            if stage == "train":
                model_.train()  # Set train mode for model (For Dropout)
                for x_batch, y_batch, _ in dataloaders[
                    stage
                ]:  # get dataloader for specific stage (train or validation)
                    x_batch, y_batch = x_batch.to(torch_device), y_batch.to(
                        torch_device
                    )  # Move to device tensors
                    y_pred = model_(x_batch)  # get pred from model
                    loss = torch.nn.functional.cross_entropy(
                        y_pred, y_batch
                    )  # compute categorical cross-entropy
                    loss_history.append(loss.item())  # append to loss_history
                    loss.backward()  # Call backward propagation on the loss
                    optimizer.step()  # Move in the parameter space
                    optimizer.zero_grad()  # set to zero gradients
            else:
                with torch.no_grad():  # we do not need gradients when calculating validation loss and accuracy
                    loss_singleval = 0  # Initialize to 0 the loss for the single iteration on the validation set
                    validation_f1 = 0
                    model_.eval()  # Evaluation mode (for dropout)
                    for x_batch, y_batch, len_masks in dataloaders[
                        stage
                    ]:  # Access the dataloader for validation
                        # Move the tensors to right device
                        x_batch, y_batch = x_batch.to(torch_device), y_batch.to(
                            torch_device
                        )
                        y_pred = model_(x_batch)  # Get prediction from validation

                        y_pred = [
                            dataloaders[stage].dataset.target_encoder.inverse_transform(
                                np.array(
                                    itertools.chain(
                                        *[
                                            itertools.repeat(x, y)
                                            for x, y in zip(prediction, len_masks)
                                        ]
                                    )
                                    for prediction in y_pred
                                )
                            )
                        ]
                        y_batch = [
                            dataloaders[stage].dataset.target_encoder.inverse_transform(
                                np.array(
                                    itertools.chain(
                                        *[
                                            itertools.repeat(x, y)
                                            for x, y in zip(label, len_masks)
                                        ]
                                    )
                                )
                            )
                            for label in y_batch
                        ]
                        y_batch = y_batch.tolist()
                        y_pred = y_pred.tolist()
                        # add loss for single batch from validation
                        loss_singleval += torch.nn.functional.cross_entropy(
                            y_pred, y_batch
                        ).item()
                        validation_f1 += f1_score(y_batch, y_pred)

                    # append mean validation loss (mean over the number of batches)
                    val_loss.append(loss_singleval / len(dataloaders[stage]))
                    seq_F1.append(validation_f1)

        p_bar.set_description(
            f'TRAIN: {sum(loss_history) / len(dataloaders["train"])};'
            f"VAL: {val_loss[-1]}; F1_VAL: {seq_F1[-1]}"
        )

    return model_, loss_history, val_loss


if __name__ == "__main__":
    # download_embeddings('small', '../../model')
    # test = load_embeddings('../../model/embeddings.txt')
    test = ModelData(
        "../../data", load_embeddings(os.path.join("../../model", "embeddings.txt"))
    )
    check = test[0]
    pass

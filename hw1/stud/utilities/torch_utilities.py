import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import json
from nltk.chunk import ne_chunk_sents
from nltk.tag import pos_tag_sents
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import numpy as np
import nltk
import itertools
from tqdm import tqdm
from copy import copy
import os
from .embeddings_utilities import load_embeddings

nltk.download("maxent_ne_chunker")
nltk.download("words")


class ModelData(Dataset):
    def __init__(self, data_folder: str, embeddings, subset: str = "train") -> None:
        self.tokens = dict()
        self.labels = dict()
        self.embeddings, self.index_emb = embeddings
        self.oov = (
            np.random.default_rng(1234)
            .normal(
                loc=self.embeddings.mean(),
                scale=self.embeddings.std(axis=1).mean(),
                size=self.embeddings.shape[1],
            )
            .astype(np.float32)
        )
        self.embeddings = np.append(self.embeddings, self.oov[np.newaxis, :], axis=0)

        tags = ["SENTIMENT", "CHANGE", "ACTION", "SCENARIO", "POSSESSION"]

        self.target_encoder = LabelEncoder().fit(
            np.array(["O"] + ["B-" + x for x in tags] + ["I-" + x for x in tags]).T
        )

        with open(os.path.join(data_folder, subset) + ".jsonl", "r") as file:
            for line in file:
                line = json.loads(line)
                self.tokens[line["idx"]] = line["tokens"]
                self.labels[line["idx"]] = line["labels"]

        pos_tagged = pos_tag_sents(self.tokens.values())
        chunked = ne_chunk_sents(pos_tagged)
        self.chunked = dict(zip(self.tokens.keys(), chunked))

    def sample_builder(self, tree: nltk.Tree) -> tuple[list[np.array], list[int]]:
        """
        The method takes care of building the actual embedding samples + a "length mask" which allows tracking the
        length of multi-token expressions which are embedded by a single vector (since they are named entities). This
        extra step is useful downstream.
        :param tree: The Tree class from NLTK containing named entity (binary) tagging.
        :return: List of embeddings + list of the lengths of the expressions embedded
        """
        tree = copy(tree)
        embeddings_ind = list()
        len_mask = list()
        while tree:
            node = tree.pop(0)
            if isinstance(node, tuple):
                len_mask.append(1)
                if node[1] == "NE":
                    key = "ENTITY/" + node[1]
                    try:
                        embeddings_ind.append(self.index_emb[key])
                        continue
                    except KeyError:
                        pass
                try:
                    embeddings_ind.append(self.index_emb[node[1].lower()])
                except KeyError:
                    embeddings_ind.append(len(self.embeddings) - 1)

            elif isinstance(node, nltk.Tree) and node.label() == "NE":
                token_set = [x[0] for x in list(node)]
                key = "ENTITY/" + "_".join([x for x in list(token_set)])
                try:
                    embeddings_ind.append(self.index_emb[key])
                    len_mask.append(len(node))
                except KeyError:
                    for token in token_set:
                        len_mask.append(1)
                        try:
                            embeddings_ind.append(self.index_emb[token.lower()])
                        except KeyError:
                            embeddings_ind.append(len(self.embeddings) - 1)

        return self.embeddings[embeddings_ind], len_mask

    def __getitem__(self, item):
        labels = self.target_encoder.transform(self.labels[item])
        tree = self.chunked[item]
        embeddings, len_mask = self.sample_builder(tree)

        grouped_labels = list()
        i = 0
        for group_len in len_mask:
            grouped_labels.append(labels[i])
            i += group_len

        return (
            torch.from_numpy(embeddings),
            torch.tensor(grouped_labels, dtype=torch.int64),
            len_mask,
        )

    def __len__(self):
        return len(self.tokens)


def obs_collate(batch):
    embeddings = [obs[0] for obs in batch]
    labels = [obs[1] for obs in batch]
    len_masks = [obs[2] for obs in batch]
    embeddings = torch.nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False)
    labels = torch.nn.utils.rnn.pack_sequence(labels, enforce_sorted=False)
    return embeddings, labels, len_masks


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
                    y_pred, len_seq = model_(x_batch)  # get pred from model
                    loss = torch.nn.functional.cross_entropy(
                        torch.swapaxes(y_pred, 1, 2),
                        torch.nn.utils.rnn.pad_packed_sequence(
                            y_batch, batch_first=True, padding_value=12
                        )[0],
                        ignore_index=12,
                    )  # compute categorical cross-entropy
                    loss_history.append(loss.item())  # append to loss_history
                    loss.backward()  # Call backward propagation on the loss
                    optimizer.step()  # Move in the parameter space
                    optimizer.zero_grad()  # set to zero gradients
            else:
                with torch.no_grad():  # we do not need gradients when calculating validation loss and accuracy
                    loss_accum = 0  # Initialize to 0 the loss for the single iteration on the validation set
                    validation_f1 = 0
                    model_.eval()  # Evaluation mode (for dropout)
                    for x_batch, y_batch, rep_masks in dataloaders[
                        stage
                    ]:  # Access the dataloader for validation
                        # Move the input tensor to right device
                        x_batch = x_batch.to(torch_device)

                        y_pred, len_seq = model_(
                            x_batch
                        )  # Get prediction from validation

                        y_batch = torch.nn.utils.rnn.pad_packed_sequence(
                            y_batch, batch_first=True, padding_value=12
                        )[0]

                        # add loss for single batch from validation
                        loss_accum += torch.nn.functional.cross_entropy(
                            torch.swapaxes(y_pred, 1, 2), y_batch, ignore_index=12
                        ).item()
                        y_pred = torch.argmin(y_pred, -1)
                        y_batch = y_batch.tolist()
                        y_batch = [
                            np.repeat(sent[:sent_len], reps).tolist()
                            for sent, sent_len, reps in zip(y_batch, len_seq, rep_masks)
                        ]
                        y_pred = y_pred.tolist()
                        y_pred = [
                            np.repeat(sent[:sent_len], reps).tolist()
                            for sent, sent_len, reps in zip(y_pred, len_seq, rep_masks)
                        ]
                        y_batch = [
                            dataloaders[stage]
                            .dataset.target_encoder.inverse_transform(
                                np.array(sent)
                            )
                            .tolist()
                            for sent in y_batch
                        ]
                        y_pred = [
                            dataloaders[stage]
                            .dataset.target_encoder.inverse_transform(
                                np.array(sent)
                            )
                            .tolist()
                            for sent in y_pred
                        ]

                        validation_f1 += f1_score(y_batch, y_pred, mode='strict', scheme=IOB2)

                    # append mean validation loss (mean over the number of batches)
                    val_loss.append(loss_accum / len(dataloaders[stage]))
                    seq_F1.append(validation_f1)

        p_bar.set_description(
            f'TRAIN: {sum(loss_history) / len(dataloaders["train"])};'
            f"VAL: {val_loss[-1]}; F1_VAL: {seq_F1[-1]}"
        )

    return model_, loss_history, val_loss


if __name__ == "__main__":
    import time

    training_data = ModelData(
        "../../../data",
        load_embeddings(os.path.join("../../../model", "embeddings.txt")),
    )
    train_dataloader = DataLoader(
        training_data, batch_size=64, shuffle=True, collate_fn=obs_collate
    )
    iterator = iter(train_dataloader)
    start = time.time()
    next(iterator)
    print(time.time() - start)

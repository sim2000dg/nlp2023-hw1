import string

import numpy as np
from typing import List
import pickle

from model import Model
from .model_def import BiLSTMClassifier
from .utilities import sample_builder
import torch
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.tag import _get_tagger, _pos_tag
from nltk.data import load
from nltk.chunk import _BINARY_NE_CHUNKER

eng_tagger = _get_tagger("eng")
ne_chunker = load(_BINARY_NE_CHUNKER)


def build_model(device: str) -> Model:
    return StudentModel(device)


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O"),
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model, BiLSTMClassifier):
    def __init__(self, device: str) -> None:
        with open("model/token_dict.pickle", "rb") as file:
            self.token_dict = pickle.load(file)
        BiLSTMClassifier.__init__(self, None, "LSTM", 500, 1, 5, 0.3)
        state_dict = torch.load("model/trained_weights_50015031e31e4.pth")
        self.load_state_dict(state_dict)
        self.device = torch.device(device)
        self.to(self.device)

        tags = [
            "SENTIMENT",
            "CHANGE",
            "ACTION",
            "SCENARIO",
            "POSSESSION",
        ]  # Set of events
        self.target_encoder = LabelEncoder().fit(
            np.array(["O"] + ["B-" + x for x in tags] + ["I-" + x for x in tags]).T
        )  # Build encoder to decode/encode labels

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        self.eval()  # Inference mode

        punct_index = [
            [i for i, x in enumerate(sentence) if x in string.punctuation]
            for sentence in tokens
        ]

        emb_indexes = list()
        rep_masks = list()
        pos_tags = list()
        for sentence in tokens:
            pos_tagged = _pos_tag(sentence, None, eng_tagger, "eng")  # Return pos tagged sentence
            ne_chunked = ne_chunker.parse(pos_tagged)  # Chunked sentence for NE recognition
            data = sample_builder(ne_chunked, self.token_dict)  # Build sample with utility func
            # Append integers
            emb_indexes.append(data[0])
            rep_masks.append(data[1])
            pos_tags.append(data[2])

        with torch.no_grad():  # We don't need to track gradient for inference
            emb_ind = torch.nn.utils.rnn.pack_sequence(
                [torch.tensor(sentence, dtype=torch.int32) for sentence in emb_indexes],
                enforce_sorted=False,
            ).to(self.device)  # Packed tensor for token embeddings
            pos_tags = torch.nn.utils.rnn.pack_sequence(
                [torch.tensor(sentence, dtype=torch.int32) for sentence in pos_tags],
                enforce_sorted=False,
            ).to(self.device)  # Packed tensor for POS tags
            preds, len_seq = self([emb_ind, pos_tags])  # Get probabilities and sentence length
            preds = torch.argmax(preds, -1)  # Get predictions
            preds = preds.tolist()  # Turn predictions into a digestible list
            preds = [x[:len_sentence] for x, len_sentence in zip(preds, len_seq)]  # Cut (padded) sentence

        decoded_preds = list()
        # Decode the predictions, add 0 label for punctuation and repeat predictions to account for NE chunking
        for sentence_pred, rep_mask, punctuation_pos in zip(
            preds, rep_masks, punct_index
        ):
            y_pred = np.repeat(sentence_pred, rep_mask)
            y_pred = self.target_encoder.inverse_transform(y_pred).tolist()
            for i in punctuation_pos:
                y_pred.insert(
                    i, "O"
                )  # O(n) operation, slow but who cares, this is a short list
            decoded_preds.append(y_pred)

        return decoded_preds

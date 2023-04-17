import string

import numpy as np
from typing import List

from model import Model
from model_def import BiLSTMClassifier
from utilities import load_embeddings, sample_builder
import torch
import nltk


nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('averaged_perceptron_tagger')

from nltk.tag import _get_tagger, _pos_tag
from nltk.data import load
from nltk.chunk import _BINARY_NE_CHUNKER

eng_tagger = _get_tagger('eng')
ne_chunker = load(_BINARY_NE_CHUNKER)



def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


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
        (463402, "O")
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
    def __init__(self):
        embedding_matrix, self.token_dict = load_embeddings('../../model/embeddings.txt')
        BiLSTMClassifier.__init__(StudentModel, embedding_matrix, 'LSTM', 200, 1, 3, 0)
        self.load_state_dict(torch.load('../../model/trained_weights.pth'))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        tokens = tokens[0]
        punct_index = [i for i, x in enumerate(tokens) if i not in string.punctuation]
        pos_tagged = _pos_tag(tokens, None, eng_tagger, 'eng')
        ne_chunked = ne_chunker(pos_tagged)
        emb_ind, rep_mask, pos_tags = sample_builder(ne_chunked, self.token_dict)
        emb_ind = torch.nn.utils.rnn.pack_sequence(torch.tensor(emb_ind, dtype=torch.int32))
        pos_tags = torch.nn.utils.rnn.pack_sequence(torch.torch.tensor(pos_tags, dtype=torch.int32))
        preds, len_seq = self([emb_ind, pos_tags])
        preds = torch.nn.utils.rnn.pad_packed_sequence(preds, batch_first=True)
        preds = torch.squeeze(preds, 0)
        preds = torch.argmax(preds, -1)
        y_pred = np.repeat(preds, rep_mask).tolist()
        for i in punct_index:
            y_pred[i] = 'O'

        return [y_pred]



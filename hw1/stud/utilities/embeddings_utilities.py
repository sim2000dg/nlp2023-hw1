import math

import requests
import bz2
import os
import shutil

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np


def download_embeddings(dimension: str, path: str) -> str:
    """
    Utility function to download the Wikipedia2Vec (no link graph) embeddings from the online repository.
    :param dimension: Dimensionality of embeddings downloaded. Small(100)/Medium(300)/Big(500) are the only choices.
    :param path: Folder where to store the embeddings.
    :return: The path where the embeddings are stored.
    """
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
    tokens = embeddings.index.str.lower().tolist()
    embeddings = embeddings.to_numpy()
    oov = (
        np.random.default_rng(1234)
        .normal(
            loc=embeddings.mean(),
            scale=embeddings.std(axis=1).mean(),
            size=embeddings.shape[1],
        )
        .astype(np.float32)
    )
    embeddings = np.append(embeddings, oov[np.newaxis, :], axis=0)

    return torch.from_numpy(embeddings), dict(zip(tokens, range(len(embeddings)-1)))

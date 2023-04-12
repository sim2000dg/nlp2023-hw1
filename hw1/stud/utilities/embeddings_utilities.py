import math

import requests
import bz2
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np


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

    return embeddings.to_numpy(), dict(zip(embeddings.index.tolist(), range(len(embeddings))))


if __name__ == "__main__":
    # download_embeddings('small', '../../model')
    test = load_embeddings('../../model/embeddings.txt')
    pass

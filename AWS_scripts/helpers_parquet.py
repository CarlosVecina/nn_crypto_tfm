# encoding: utf-8
from itertools import repeat
from os import listdir
from os.path import join

import pyarrow.parquet as pq


def todo_parquet(path):
    dataset = pq.ParquetDataset(path)
    return dataset.read().to_pandas()


def batches_parquet(path, batch_size=100):
    dataset = pq.ParquetDataset(path)
    yield from dataset.read().to_batches(batch_size)


def folder_parquet(dirpath, dias_validacion=2):
    def generator_paths(paths):
        for filename in paths:
            path = join(dirpath, filename)
            yield todo_parquet(path)

    paths = tuple(x for x in listdir(dirpath) if x != "_SUCCESS")
    paths = sorted(paths)
    return generator_paths(paths[:-dias_validacion]), generator_paths(paths[-dias_validacion:])

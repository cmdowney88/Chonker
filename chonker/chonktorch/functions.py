"""A module of functions for data pipelines using PyTorch"""

import json
import math
import random
from logging import Logger

import torch
import numpy as np
import torch.nn as nn

from ..wrangle import Vocab


def sort_unsort_pmts(lst, descending=False):
    """
    Return the permutation used to sort a list, as well as the permutation
    necessary to restore it to its original order
    """

    sort_zip = list(zip(lst, [x for x in range(len(lst))]))
    sort_zip.sort(key=lambda x: x[0], reverse=descending)
    sort_pmt = [x[1] for x in sort_zip]
    unsort_zip = list(zip(sort_pmt, [x for x in range(len(sort_pmt))]))
    unsort_zip.sort(key=lambda x: x[0])
    unsort_pmt = [x[1] for x in unsort_zip]
    return sort_pmt, unsort_pmt


def shuffle_unshuffle_pmts(length):
    """
    Return a random shuffle permutation over a given list length, as well as the
    permutation necessary to restore it to its original order
    """

    shuffle_pmt = [x for x in range(length)]
    random.shuffle(shuffle_pmt)
    unshuffle_zip = list(zip(shuffle_pmt, [x for x in range(length)]))
    unshuffle_zip.sort(key=lambda x: x[0])
    unshuffle_pmt = [x[1] for x in unshuffle_zip]
    return shuffle_pmt, unshuffle_pmt


def partition(data, seq_len, batch_size):
    """
    Partition the data into batches of equal sequence lengths for parallel
    operation
    
    Takes the batch size from parameters, along with sequence length, and
    determines the shape of the training tensor. For instance, if the number of
    batch size is B and the sequnce length is L, the tensor will be of shape
    (X, B) such that X is divisible by L
    """

    denominator = seq_len * batch_size
    num_batches = math.floor(len(data) / denominator)
    new_length = num_batches * denominator
    data_tensor = torch.tensor(data[:new_length])
    data_partitions = data_tensor.view(batch_size, -1).t().contiguous()
    assert data_partitions.size(0) % seq_len == 0
    return data_partitions, new_length


def pad_and_batch(
    data,
    batch_size,
    padding_value,
    gradient_accumulation=1,
    batch_shuffling="none"
):
    """
    Sort input sequences into batches based on sequence length, padding as
    necessary, and optionally shuffling by batch

    Returns a dataset consisting of the batched tensor, sequence lengths, and
    the permutations used to sort/shuffle and un-sort/shuffle the data. All
    sequences must be sorted into descending order for computational
    compatibility, and batches can optionally be sorted. Final shape of the
    tensor will be (num_batches, max_seq_length, batch_size). Final shape of
    the lengths index will be (num_batches, batch_size)
    """

    lengths = [len(line) for line in data]
    full_batch_size = batch_size * gradient_accumulation
    num_full_batches = math.floor(len(data) / full_batch_size)
    data = data[:num_full_batches * full_batch_size]
    lengths = lengths[:num_full_batches * full_batch_size]
    num_batches = num_full_batches * gradient_accumulation

    sort_pmt, unsort_pmt = sort_unsort_pmts(lengths, descending=True)
    data = [data[i] for i in sort_pmt]
    lengths = [lengths[i] for i in sort_pmt]

    data = [torch.tensor(line) for line in data]
    tensor = nn.utils.rnn.pad_sequence(data, padding_value=padding_value)
    max_seq_len = tensor.size(0)
    tensor = tensor.t().view(-1, batch_size, max_seq_len).transpose(1, 2)
    lengths = list(np.array_split(np.array(lengths), num_batches))
    if batch_shuffling != "none":
        shuffle_pmt, unshuffle_pmt = shuffle_unshuffle_pmts(len(tensor))
        tensor = tensor[shuffle_pmt, :, :]
        lengths = [lengths[i] for i in shuffle_pmt]
        data_set = {
            'tensor': tensor.contiguous(),
            'lengths': lengths,
            'sort_pmt': sort_pmt,
            'unsort_pmt': unsort_pmt,
            'batch_shuffle_pmt': shuffle_pmt,
            'batch_unshuffle_pmt': unshuffle_pmt
        }
    else:
        data_set = {
            'tensor': tensor.contiguous(),
            'lengths': lengths,
            'sort_pmt': sort_pmt,
            'unsort_pmt': unsort_pmt
        }
    return data_set


def import_embeddings(
    embedding_path: str,
    vocab: Vocab,
    indices_path: str = None,
    init_range: float = 1.0,
    logger: Logger = None
):
    """
    Import pretrained NumPy embeddings for a given vocab, adding a randomized 
    embedding if a vocabulary item is not included in the pretrained keys
    
    Args:
        embedding_path: The base path to the pretrained embeddings. Assumes that
            the numpy embeddings are at `{embedding_path}.npy`. If 
            `indices_path` is `None`, assumes the embedding indices are at 
            `{embedding_path}_indices.json`
        vocab: The Vocab object for which to gather pretrained embeddings.
            Embeddings not correponding to vocab items are not imported
        indices_path: The path to the embedding indices. If `None`, defaults to
            `{embedding_path}_indices.json`. Default: `None`
        init_range: The positive end of the range around zero with which to
            initialize missing embeddings. Default: `1.0`
        logger: A logger through which to pipe informational messages
    Returns:
    """
    if logger:
        logger.info(
            'Importing pretrained character embeddings from'
            f' {embedding_path}.npy'
        )
    embeddings = np.load(f"{embedding_path}.npy")
    if not indices_path:
        indices_path = f"{embedding_path}_indices.json"
    with open(indices_path, "r") as f:
        embedding_tok_to_id = json.load(f)
    embedding_dim = embeddings.shape[1]
    sorted_embeddings = []
    randomized_embeddings = []
    for idx in sorted(vocab.id_to_tok.keys()):
        token = vocab.id_to_tok[idx]
        if token in embedding_tok_to_id:
            sorted_embeddings.append(embeddings[embedding_tok_to_id[token]])
        else:
            sorted_embeddings.append(
                np.random.uniform(-init_range, init_range, (embedding_dim))
            )
            randomized_embeddings.append(token)
    pretrained_embeddings = np.array(sorted_embeddings)
    assert pretrained_embeddings.shape[0] == len(vocab.id_to_tok)
    assert pretrained_embeddings.shape[1] == embedding_dim
    if logger:
        logger.info(
            'Tokens without pretrained embeddings:'
            f' {randomized_embeddings}, randomly initializing'
        )
    return pretrained_embeddings


def _lr_lambda(
    total_num_steps,
    num_warmup_steps=0,
    warmup='linear',
    decay='linear',
    gamma=0.9,
    gamma_steps=1000
):

    if warmup == 'flat':
        warmup_lambda = lambda step: 1.0
    elif warmup == 'linear':
        warmup_lambda = lambda step: (step + 1) / (num_warmup_steps)
    else:
        raise ValueError(f'Warmup mode {warmup} is not valid')

    if decay == 'linear':
        decay_lambda = lambda step: (
            (total_num_steps - (step + 1)) /
            (total_num_steps - num_warmup_steps)
        )
    elif decay == 'exponential':
        decay_lambda = lambda step: (
            gamma**((step + 1 - num_warmup_steps) / gamma_steps)
        )
    else:
        raise ValueError(f'Decay mode {decay} is not valid')

    lr_lambda = lambda step: (
        warmup_lambda(step) if step < num_warmup_steps else decay_lambda(step)
    )
    return lr_lambda


def get_lr_lambda_by_epoch(
    num_epochs,
    batches_per_epoch,
    num_warmup_epochs=1,
    warmup='linear',
    decay='linear',
    gamma=0.9,
    gamma_epochs=1
):
    """
    Get learning-rate step lambda based on the total nummber of epochs,
    batches per epoch, and number of warmup epochs

    Warmup can be `flat` or `linear`. Decay can be `linear` or `exponential`.
    Exponential decay is defined by the `gamma` base and `gamma_epochs`
    period over which the decay applies. For instance, if `gamma` is 0.5 and
    `gamma_epochs` is 1, the learning rate will decay by half every epoch
    """

    total_num_steps = num_epochs * batches_per_epoch
    num_warmup_steps = num_warmup_epochs * batches_per_epoch
    gamma_steps = gamma_epochs * batches_per_epoch

    return _lr_lambda(
        total_num_steps,
        num_warmup_steps=num_warmup_steps,
        warmup=warmup,
        decay=decay,
        gamma=gamma,
        gamma_steps=gamma_steps
    )


def get_lr_lambda_by_steps(
    total_num_steps,
    num_warmup_steps=0,
    warmup='linear',
    decay='linear',
    gamma=0.9,
    gamma_steps=1000
):
    """
    Get learning-rate step lambda based on the total nummber of steps and the
    number of warmup steps

    Warmup can be `flat` or `linear`. Decay can be `linear` or `exponential`.
    Exponential decay is defined by the `gamma` base and `gamma_steps` period
    over which the decay applies. For instance, if `gamma` is 0.5 and
    `gamma_steps` is 1000, the learning rate will decay by half every 1000 steps
    """

    return _lr_lambda(
        total_num_steps,
        num_warmup_steps=num_warmup_steps,
        warmup=warmup,
        decay=decay,
        gamma=gamma,
        gamma_steps=gamma_steps
    )

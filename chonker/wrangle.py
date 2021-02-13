"""
A module containing convenience classes and functions for the pre-processing 
of data being fed into statistical and machine learning models
"""

import copy
import itertools
import re
import yaml


def get_lines(file):
    """
    Return all lines from a file stripping newline characters and ignoring
    empty lines
    """
    all_lines = [line.strip('\n') for line in open(file, 'r')]
    return [line for line in all_lines if line != '']


def tokenize_tags(string):
    """Tokenize <tags> away from surrounding non-whitespace characters"""
    string = re.sub(r"(<[A-Za-z0-9]*>)(?=<[A-Za-z0-9]*>)", r"\1 ", string)
    string = re.sub(r"(\S+)(<[A-Za-z0-9]*>)(\s+|$)", r"\1 \2\3", string)
    string = re.sub(r"(^|\s+)(<[A-Za-z0-9]*>)(\S+)", r"\1\2 \3", string)
    return re.sub(r"(?=\S+)(<[A-Za-z0-9]*>)(?=\S+)", r" \1 ", string)


def split_lines(lines, delimiter=r'\s+', split_tags=False):
    """Split a list of strings based on a regex delimiter"""
    if split_tags:
        lines = [tokenize_tags(line) for line in lines]
    return [re.split(delimiter, line) for line in lines]


def flatten(multi_list):
    """Flatten a list of lists into a single list"""
    return list(itertools.chain.from_iterable(multi_list))


def is_tag(word):
    """Define a word as a tag if it starts and ends with angle brackets"""
    return word.startswith('<') and word.endswith('>')


def case(string, preserve=False):
    if preserve:
        return string
    else:
        return string.lower()


def chars_from_words(sequence):
    """
    From a list of tokenized words, return the same list with the words
    sub-tokenized into characters, unless the word is a tag
    """
    return [
        [char for char in word] if not is_tag(word) else [word]
        for word in sequence
    ]


def basic_tokenize(
    in_file,
    preserve_case=False,
    split_tags=False,
    edge_tokens=False,
    out_file=None
):
    """
    Whitespace tokenize the input file, convert to lowercase and return the 
    tokenized text

    If an `out_file` is given, print the tokenized text to this file
    """
    text = split_lines(get_lines(in_file), split_tags=split_tags)
    text = [[case(word, preserve_case) for word in line] for line in text]
    if edge_tokens:
        text = [['<bos>'] + line + ['<eos>'] for line in text]
    if out_file:
        with open(out_file, 'w') as f:
            for line in text:
                print(' '.join(line), file=f)
    else:
        return text


def character_tokenize(
    in_file,
    preserve_case=False,
    split_tags=True,
    edge_tokens=False,
    out_file=None
):
    """
    Character tokenize the input file, lowercasing and exlcuding whitespace
    
    If an `out_file` is given print the tokenized text to this file
    """
    text = split_lines(get_lines(in_file), split_tags=split_tags)
    text = [
        flatten(
            [
                [char for char in case(word, preserve_case)]
                if not is_tag(word) else [case(word, preserve_case)]
                for word in line
            ]
        ) for line in text
    ]
    if edge_tokens:
        text = [['<bos>'] + line + ['<eos>'] for line in text]
    if out_file:
        with open(out_file, 'w') as f:
            for line in text:
                print(''.join(line), file=f)
    else:
        return text


def get_ngrams(corpus, max_length, min_count=1):
    """
    Get 1-to-n-grams over a corpus of input sentences, discarding entries if
    they occur less than `min_count` times, as well as bidrectional dictionary
    of ngrams and indices
    """
    counts = {}

    for n in range(1, max_length + 1):
        for sentence in corpus:
            sentence_length = len(sentence)
            num_ngrams = sentence_length + 1 - n
            for i in range(num_ngrams):
                ngram = tuple(sentence[i:i + n])
                if ngram not in counts:
                    counts[ngram] = 1
                else:
                    counts[ngram] += 1

    to_delete = []
    for ngram in counts.keys():
        if counts[ngram] < min_count:
            to_delete.append(ngram)
    for ngram in to_delete:
        del counts[ngram]

    ngram_to_id = {}
    id_to_ngram = {}
    for ngram in counts.keys():
        index = len(ngram_to_id)
        ngram_to_id[ngram] = index
        id_to_ngram[index] = ngram

    return (counts, (ngram_to_id, id_to_ngram))


class Vocab():
    """
    A bi-directional mapping between the string tokens and integer IDs, 
    initialized from a text in the form of a list of lists
    """
    def __init__(self, source=None, unk_token="<unk>", other_tokens=None):
        """Initialize `Vocab` object from a text"""

        self.unk_token = unk_token

        self.tok_to_id = {}
        self.id_to_tok = {}
        self.add_vocab([self.unk_token])
        if other_tokens:
            self.add_vocab(other_tokens)
        self.unk_id = self.tok_to_id[self.unk_token]

        self.processed_sources = []
        if source:
            self.add_source(source)

    def add_vocab(self, tokens):
        """Add vocabulary items directly as a string or list of strings"""
        if type(tokens) != list:
            tokens = [tokens]
        for token in tokens:
            if token not in self.tok_to_id:
                index = len(self.tok_to_id)
                self.tok_to_id[token] = index
                self.id_to_tok[index] = token

    def add_source(self, source):
        """
        Iterate through the source texts in `to_tokenize` If they have not 
        yet been input, tokenize them and add their vocabulary
        """
        if source not in self.processed_sources:
            tokens = list(itertools.chain.from_iterable(source))
            for token in tokens:
                self.add_vocab(token)
            self.processed_sources.append(source)

    def source_added(self, source):
        """Return new copy of `Vocab` object with new source text added"""
        new_vocab = copy.deepcopy(self)
        new_vocab.add_source(source)
        return new_vocab

    def reset(self):
        """Completely empty all vocabulary items and source file lists"""
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.processed_sources = []

    def size(self):
        """Return the size of the vocabulary (number of unique IDs)"""
        size1 = len(self.tok_to_id)
        size2 = len(self.id_to_tok)
        assert size1 == size2
        return size1

    def to_ids(self, tokens):
        """
        Take in a string or list of string tokens and return the list of their 
        corresponding integer IDs
        """
        if type(tokens) != list:
            tokens = [tokens]
        output = []
        for token in tokens:
            if token not in self.tok_to_id:
                output.append(self.unk_id)
            else:
                output.append(self.tok_to_id[token])
        return output

    def to_tokens(self, ids):
        """
        Take in an integer ID or list of IDs and return the list of their 
        corresponding string tokens
        """
        if type(ids) != list:
            ids = [ids]
        output = []
        for item in ids:
            if item not in self.id_to_tok:
                output.append(self.unk_token)
            else:
                output.append(self.id_to_tok[item])
        return output

    def save(self, out_file):
        """
        Save the `Vocab` mapping to a yaml file

        By default, the yaml object will be sorted by key (the token id). NOTE:
        `processed_sources` attribute will not be saved, and so cannot be
        recovered when loading from a saved `Vocab`
        """
        with open(out_file, 'w') as f:
            yaml.dump(self.id_to_tok, f)

    def load(self, in_file):
        """
        Load the `Vocab` mapping from a saved yaml file

        NOTE: loading a saved Vocab will reset the current object, including any
        special tokens. The unk token is always presumed to be at index 0
        """
        self.processed_sources = []
        self.unk_id = None
        self.unk_token = None
        self.tok_to_id = {}
        with open(in_file, 'r') as f:
            self.id_to_tok = yaml.load(f, Loader=yaml.SafeLoader)
        for id in self.id_to_tok:
            self.tok_to_id[self.id_to_tok[id]] = id
        self.unk_id = 0
        self.unk_token = self.id_to_tok[0]

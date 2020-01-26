'''
A module containing convenience classes and functions for the pre-processing 
of data being fed into statistical and machine learning models
'''

import re
import itertools
import copy
import yaml


def get_lines(file):
    all_lines = [line.strip('\n') for line in open(file, 'r')]
    return [line for line in all_lines if line != '']


def split_lines(lines, delimiter=r'\s+'):
    return [re.split(delimiter, line) for line in lines]


def flatten(multi_list):
    return list(itertools.chain.from_iterable(multi_list))


def is_tag(word):
    return word.startswith('<') and word.endswith('>')


def chars_from_words(sequence):
    return [
        [char for char in word] if not is_tag(word) else [word]
        for word in sequence
    ]


def basic_tokenize(in_file, out_file=None):
    '''
    Whitespace tokenize the input file, convert to lowercase and return the 
    tokenized text. If an `out_file` is given, print the tokenized text to 
    this file
    '''
    text = split_lines(get_lines(in_file))
    text = [
        [word.lower() for word in line] 
        for line in text
    ]
    if out_file:
        with open(out_file, 'w') as f:
            for line in text:
                print(' '.join(line), file=f)
    else:
        return text


def character_tokenize(in_file, out_file=None):
    '''
    Character tokenize the input file, lowercasing and exlcuding whitespace. 
    If an `out_file` is given print the tokenized text to this file
    '''
    text = split_lines(get_lines(in_file))
    text = [
        flatten(
            [
                [char for char in word.lower()] 
                if not is_tag(word) 
                else [word.lower()] 
                for word in line
            ]
        ) 
        for line in text
    ]
    if out_file:
        with open(out_file, 'w') as f: 
            for line in text:
                print(''.join(line), file=f)
    else:
        return text


class Vocab():
    '''
    A bi-directional mapping between the string tokens and integer IDs, 
    initialized from a text in the form of a list of lists
    '''
    
    def __init__(self, 
                 source=None,
                 unk_token="<unk>",
                 other_tokens=None):
        '''Initialize `Vocab` object from a text'''

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
        '''Add vocabulary items directly as a string or list of strings'''
        if type(tokens) != list: tokens = [tokens]
        for token in tokens:
            if token not in self.tok_to_id:
                index = len(self.tok_to_id)
                self.tok_to_id[token] = index
                self.id_to_tok[index] = token

    def add_source(self, source):
        '''
        Iterate through the source texts in `to_tokenize`. If they have not 
        yet been input, tokenize them and add their vocabulary
        '''
        if source not in self.processed_sources:
            tokens = list(itertools.chain.from_iterable(source))
            for token in tokens:
                self.add_vocab(token)
            self.processed_sources.append(source)

    def source_added(self, source):
        '''Return new copy of `Vocab` object with new source text added'''
        new_vocab = copy.deepcopy(self)
        new_vocab.add_source(source)
        return new_vocab

    def reset(self):
        '''Completely empty all vocabulary items and source file lists'''
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.processed_sources = []

    def size(self):
        '''Return the size of the vocabulary (number of unique IDs)'''
        size1 = len(self.tok_to_id)
        size2 = len(self.id_to_tok)
        assert size1 == size2
        return size1

    def to_ids(self, tokens):
        '''
        Take in a string or list of string tokens and return the list of their 
        corresponding integer IDs
        '''
        if type(tokens) != list: tokens = [tokens]
        output = []
        for token in tokens:
            if token not in self.tok_to_id:
                output.append(self.unk_id)
            else:
                output.append(self.tok_to_id[token])
        return output

    def to_tokens(self, ids):
        '''
        Take in an integer ID or list of IDs and return the list of their 
        corresponding string tokens
        '''
        if type(ids) != list: ids = [ids]
        output = []
        for item in ids:
            if item not in self.id_to_tok:
                output.append(self.unk_token)
            else:
                output.append(self.id_to_tok[item])
        return output

    def save(self, out_file):
        '''
        Save the `Vocab` mapping to a yaml file. By default, the yaml object
        will be sorted by key (the token id). NOTE: `processed_sources`
        attribute will not be saved, and so cannot be recovered when loading
        from a saved `Vocab`
        '''
        with open(out_file, 'w') as f:
            yaml.dump(self.id_to_tok, f)
    
    def load(self, in_file):
        '''
        Load the `Vocab` mapping from a saved yaml file. NOTE: loading a saved
        Vocab will reset the current object, including any special tokens. The
        unk token is always presumed to be at index 0
        '''
        self.processed_sources = []
        self.unk_id = None
        self.unk_token = None
        self.tok_to_id = {}
        with open(in_file, 'r') as f:
            self.id_to_tok = yaml.load(f)
        for id in self.id_to_tok:
            self.tok_to_id[self.id_to_tok[id]] = id
        self.unk_id = 0
        self.unk_token = self.id_to_tok[0]

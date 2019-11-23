'''
A module containing convenience classes and functions for the pre-processing of data being fed into
statistical and Machine Learning models (i.e. "data wrangling")
'''

import re
import itertools
import copy

class Vocab():
    '''
    A bi-directional mapping between the string tokens and integer IDs, initialized from a text file
    '''
    
    def __init__(self, 
                 source_files, 
                 delimiter=r'\s+', 
                 unk_token="<UNK>", 
                 bos_token="<BOS>", 
                 eos_token="<EOS>",
                 other_tokens=None):
        '''Initialize `Vocab` object from a source file'''

        self.delimiter = delimiter
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.tok_to_id = {}
        self.id_to_tok = {}
        self.add_vocab([self.unk_token, self.bos_token, self.eos_token])
        if other_tokens:
            self.add_vocab(other_tokens)
        self.unk_id = self.tok_to_id[self.unk_token]
        self.bos_id = self.tok_to_id[self.bos_token]
        self.eos_id = self.tok_to_id[self.eos_token]
        self.source_texts = []

        if type(source_files) != list: source_files = [source_files]
        self.to_tokenize = source_files
        self.tokenized_sources = []
        self.refreshed = False
        
        self.tokenized_sources.append(self._refresh())

    def _refresh(self):
        '''
        Iterate through the source files in `to_tokenize`. If they have not yet been input, 
        tokenize them and add their vocabulary
        '''
        tokenized_texts = []
        for file in self.to_tokenize:
            if file not in self.tokenized_sources:
                with open(file, 'r') as f:
                    lines = f.readlines()
                tokenized_lines = [[tok for tok in re.split(self.delimiter, line.strip()) if tok != ''] for line in lines]
                tokens = list(itertools.chain.from_iterable(tokenized_lines))
                for token in tokens:
                    self.add_vocab(token)
                self.tokenized_sources.append(file)
                tokenized_texts.append(tokenized_lines)
        self.to_tokenize = []
        self.refreshed = True
        return tokenized_texts

    def add_vocab(self, tokens):
        '''Add vocabulary items directly as a string or list of strings'''
        if type(tokens) != list: tokens = [tokens]
        for token in tokens:
            if token not in self.tok_to_id:
                index = len(self.tok_to_id)
                self.tok_to_id[token] = index
                self.id_to_tok[index] = token

    def add_sources(self, files):
        '''Add file or list of files to the `to_tokenize` list and then refresh vocabulary'''
        if files is not list: files = [files]
        for file in files:
            self.to_tokenize.append(file)
        self.source_texts.append(self._refresh())

    def sources_added(self, files):
        '''Return new copy of `Vocab` object with new source files added'''
        new_voc = copy.deepcopy(self)
        new_voc.add_sources(files)
        return new_voc

    def reset(self):
        '''Completely empty all vocabulary items and source file lists'''
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.to_tokenize = []
        self.tokenized_sources = []
        self.source_texts = []
        self.refreshed = True

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
    
    def size(self):
        '''Return the size of the vocabulary (number of unique IDs)'''
        size1 = len(self.tok_to_id)
        size2 = len(self.id_to_tok)
        assert size1 == size2
        return size1

    def save(self, out_file, explicit_id=False):
        '''
        Save the `Vocab` mapping to a text file. The ID of the token corresponds to the line 
        on which it is printed (starting at zero). If `explicit_id` is set to `True`, print the 
        ID after the token, separated by a space.
        '''
        with open(out_file, 'w') as f:
            items = [(key, self.tok_to_id[key]) for key in self.tok_to_id]
            items.sort(key=lambda x: x[1])
            if explicit_id:
                for item in items:
                    print(f"{item[0]} {item[1]}", file=f)
            else:
                for item in items:
                    print(item[0], file=f)


def flatten(multi_list):
    return list(itertools.chain.from_iterable(multi_list))


def get_lines(file):
    return [line.rstrip('\n') for line in open(file, 'r')]


def split_lines(lines, delimiter=r'\s+'):
    return [re.split(delimiter, line) for line in lines]


def basic_tokenize(in_file, out_file=None):
    '''
    Whitespace tokenize the input file, convert to lowercase and return the tokenized text. If an 
    `out_file` is given, print the tokenized text to this file.
    '''
    text = [line.rstrip('\n') for line in open(in_file, 'r')]
    text = [re.split(r'\s+', line) for line in text]
    text = [[word.lower() for word in line] for line in text]
    if out_file:
        with open(out_file, 'w') as f:
            for line in text:
                print(' '.join(line), file=f)
    else:
        return text


def character_tokenize(in_file, out_file=None):
    '''
    Character tokenize the input file, lowercasing and exlcuding whitespace. If an `out_file`
    is given print the tokenized text to this file.
    '''
    text = [re.split(r'\s+', line.rstrip('\n')) for line in open(in_file, 'r')]
    text = [list(itertools.chain.from_iterable([[char for char in word.lower()] for word in line])) for line in text]
    if out_file:
        with open(out_file, 'w') as f: 
            for line in text:
                print(''.join(line), file=f)
    else:
        return text

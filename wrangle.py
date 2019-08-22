import sys
import os
import re
import itertools
import copy
from nltk import word_tokenize

class Vocab():
    def __init__(self, 
        source_files, 
        delimiter=r'\s+', 
        unk_token="<UNK>", 
        bos_token="<BOS>", 
        eos_token="<EOS>"):
        
        self.delimiter = delimiter
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.tok_to_id = {}
        self.id_to_tok = {}
        self.add_vocab([self.unk_token, self.bos_token, self.eos_token])
        self.unk_id = self.tok_to_id[self.unk_token]
        self.bos_id = self.tok_to_id[self.bos_token]
        self.eos_id = self.tok_to_id[self.eos_token]

        if type(source_files) != list: source_files = [source_files]
        self.to_tokenize = source_files
        self.tokenized_sources = []
        self.refreshed = False
        
        self._refresh()

    def _refresh(self):
        for file in self.to_tokenize:
            if file not in self.tokenized_sources:
                with open(file, 'r') as f:
                    lines = f.readlines()
                tokenized_lines = [re.split(self.delimiter, line.strip()) for line in lines]
                tokens = list(itertools.chain.from_iterable(tokenized_lines))
                tokens = [tok for tok in tokens if tok != '']
                for token in tokens:
                    self.add_vocab(token)
                self.tokenized_sources.append(file)
        self.to_tokenize = []
        self.refreshed = True

    def add_vocab(self, tokens):
        if type(tokens) != list: tokens = [tokens]
        for token in tokens:
            if token not in self.tok_to_id:
                index = len(self.tok_to_id)
                self.tok_to_id[token] = index
                self.id_to_tok[index] = token

    def add_sources(self, files):
        if files is not list: files = [files]
        for file in files:
            if file not in self.tokenized_sources:
                self.refreshed = False
                self.to_tokenize.append(file)
        self._refresh()

    def sources_added(self, files):
        new_voc = copy.deepcopy(self)
        new_voc.add_sources(files)
        return new_voc

    def reset(self):
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.to_tokenize = []
        self.tokenized_sources = []
        self.refreshed = True

    def to_ids(self, tokens):
        if type(tokens) != list: tokens = [tokens]
        output = []
        for token in tokens:
            if token not in self.tok_to_id:
                output.append(self.unk_id)
            else:
                output.append(self.tok_to_id[token])
        return output

    def to_tokens(self, ids):
        if type(ids) != list: ids = [ids]
        output = []
        for item in ids:
            if item not in self.id_to_tok:
                output.append(self.unk_token)
            else:
                output.append(self.id_to_tok[item])
        return output
    
    def size(self):
        size1 = len(self.tok_to_id)
        size2 = len(self.id_to_tok)
        assert size1 == size2
        return size1

    def save(self, out_file, explicit_id=False):
        with open(out_file, 'w') as f:
            items = [(key, self.tok_to_id[key]) for key in self.tok_to_id]
            items.sort(key=lambda x: x[1])
            if explicit_id:
                for item in items:
                    print(f"{item[0]} {item[1]}", file=f)
            else:
                for item in items:
                    print(item[0], file=f)



def basic_tokenize(in_file, out_file):
    text = [line.rstrip('\n') for line in open(in_file, 'r')]
    text = [word_tokenize(line) for line in text]
    text = [[word.lower() for word in line] for line in text]
    with open(out_file, 'w') as f:
        for line in text:
            print(' '.join(line), file=f)

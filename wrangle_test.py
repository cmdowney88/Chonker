import pytest
import copy
import os
from chonker import wrangle as wr

lines = wr.get_lines('test/test_lines.txt')
char_line_raw = 'Thisisastringofcharacters'
space_line_raw = 'These words  are <TAG> separated by spaces'
tab_line_raw = 'These\twords\tare\ttab\tdelimited'

space_line_split = [
    'These', 'words', 'are', '<TAG>', 'separated', 'by', 'spaces'
]
tab_line_split = ['These', 'words', 'are', 'tab', 'delimited']
tag_line_split = [
    '<PUNC>These', 'are', '<PUNC>edge', 'cases', 'for', 'tags<PUNC>',
    '1<PUNC>2', '<CUR>8<PUNC><PUNC><PUNC>'
]


def test_get_lines():
    assert lines[0] == char_line_raw
    assert lines[1] == space_line_raw
    assert lines[2] == tab_line_raw


ws_split_lines = wr.split_lines(lines)
space_split_lines = wr.split_lines(lines, delimiter=r'\ +')
tab_split_lines = wr.split_lines(lines, delimiter=r'\t+')


def test_ws_split_lines():
    assert ws_split_lines[0] == [char_line_raw]
    assert ws_split_lines[1] == space_line_split
    assert ws_split_lines[2] == tab_line_split


def test_space_split_lines():
    assert space_split_lines[0] == [char_line_raw]
    assert space_split_lines[1] == space_line_split
    assert space_split_lines[2] == [tab_line_raw]


def test_tab_split_lines():
    assert tab_split_lines[0] == [char_line_raw]
    assert tab_split_lines[1] == [space_line_raw]
    assert tab_split_lines[2] == tab_line_split


flattened_matrix = wr.flatten(ws_split_lines)
concat_lines = [
    char_line_raw
] + space_line_split + tab_line_split + tag_line_split


def test_flatten():
    assert flattened_matrix == concat_lines


basic_tokenized_text = wr.basic_tokenize('test/test_lines.txt')
char_line_tok = ['thisisastringofcharacters']
space_line_tok = ['these', 'words', 'are', '<tag>', 'separated', 'by', 'spaces']
tab_line_tok = ['these', 'words', 'are', 'tab', 'delimited']
tag_line_tok = [
    '<punc>these', 'are', '<punc>edge', 'cases', 'for', 'tags<punc>',
    '1<punc>2', '<cur>8<punc><punc><punc>'
]


def test_basic_tokenize():
    assert basic_tokenized_text == [
        char_line_tok, space_line_tok, tab_line_tok, tag_line_tok
    ]


basic_tokenized_with_edges = wr.basic_tokenize(
    'test/test_lines.txt', edge_tokens=True
)
tok_target_with_edges = [
    ['<bos>'] + line + ['<eos>']
    for line in [char_line_tok, space_line_tok, tab_line_tok, tag_line_tok]
]


def test_tokenize_with_edges():
    assert basic_tokenized_with_edges == tok_target_with_edges
    print(basic_tokenized_with_edges)


character_tokenized_text = wr.character_tokenize('test/test_lines.txt')
char_line_chars = [char for char in 'thisisastringofcharacters']
space_line_chars = [char for char in ''.join(space_line_tok[:3])]
space_line_chars += ['<tag>'] + [char for char in ''.join(space_line_tok[4:])]
tag_line_chars = [
    '<punc>', 't', 'h', 'e', 's', 'e', 'a', 'r', 'e', '<punc>', 'e', 'd', 'g',
    'e', 'c', 'a', 's', 'e', 's', 'f', 'o', 'r', 't', 'a', 'g', 's', '<punc>',
    '1', '<punc>', '2', '<cur>', '8', '<punc>', '<punc>', '<punc>'
]
tab_line_chars = [char for char in ''.join(tab_line_tok)]


def test_character_tokenize():
    assert character_tokenized_text[0] == char_line_chars
    assert character_tokenized_text[1] == space_line_chars
    assert character_tokenized_text[2] == tab_line_chars
    assert character_tokenized_text[3] == tag_line_chars


chars_from_space_line = [
    ['t', 'h', 'e', 's', 'e'], ['w', 'o', 'r', 'd', 's'], ['a', 'r', 'e'],
    ['<tag>']
]


def test_chars_from_words():
    assert wr.chars_from_words(space_line_tok[:4]) == chars_from_space_line


vocab = wr.Vocab(basic_tokenized_text)
tok_to_id_1 = {
    '<unk>': 0,
    'thisisastringofcharacters': 1,
    'these': 2,
    'words': 3,
    'are': 4,
    '<tag>': 5,
    'separated': 6,
    'by': 7,
    'spaces': 8,
    'tab': 9,
    'delimited': 10,
    '<punc>these': 11,
    '<punc>edge': 12,
    'cases': 13,
    'for': 14,
    'tags<punc>': 15,
    '1<punc>2': 16,
    '<cur>8<punc><punc><punc>': 17
}
id_to_tok_1 = {
    0: '<unk>',
    1: 'thisisastringofcharacters',
    2: 'these',
    3: 'words',
    4: 'are',
    5: '<tag>',
    6: 'separated',
    7: 'by',
    8: 'spaces',
    9: 'tab',
    10: 'delimited',
    11: '<punc>these',
    12: '<punc>edge',
    13: 'cases',
    14: 'for',
    15: 'tags<punc>',
    16: '1<punc>2',
    17: '<cur>8<punc><punc><punc>'
}


def test_vocab_init():
    assert vocab.tok_to_id == tok_to_id_1
    assert vocab.id_to_tok == id_to_tok_1
    assert vocab.processed_sources == [basic_tokenized_text]


def test_vocab_add_vocab():
    voc = copy.deepcopy(vocab)
    voc.add_vocab('cat')
    voc.add_vocab(['kitty', 'cat'])
    assert voc.tok_to_id['cat'] == 18
    assert voc.id_to_tok[18] == 'cat'
    assert voc.tok_to_id['kitty'] == 19
    assert voc.id_to_tok[19] == 'kitty'
    assert len(voc.tok_to_id) == len(voc.id_to_tok) == 20


new_source = [['this', 'is', 'a', 'cat'], ['these', 'are', 'cats']]


def test_vocab_add_source():
    voc = copy.deepcopy(vocab)
    voc.add_source(basic_tokenized_text)
    assert voc.processed_sources == vocab.processed_sources
    assert vocab.processed_sources == [basic_tokenized_text]
    voc.add_source(new_source)
    assert voc.processed_sources == [basic_tokenized_text, new_source]


def test_vocab_source_added():
    voc = vocab.source_added(new_source)
    assert vocab.processed_sources == [basic_tokenized_text]
    assert voc.processed_sources == [basic_tokenized_text, new_source]


def test_vocab_reset():
    voc = copy.deepcopy(vocab)
    voc.reset()
    assert len(voc.tok_to_id) == len(voc.id_to_tok) == 0


def test_vocab_size():
    assert vocab.size() == len(vocab.tok_to_id) == len(vocab.id_to_tok)


def test_vocab_to_ids():
    assert vocab.to_ids('<tag>') == [5]
    assert vocab.to_ids(['these', 'are', 'spaces']) == [2, 4, 8]
    assert vocab.to_ids(['these', 'are', 'cats']) == [2, 4, 0]


def test_vocab_to_tokens():
    assert vocab.to_tokens(5) == ['<tag>']
    assert vocab.to_tokens([2, 4, 8]) == ['these', 'are', 'spaces']
    cat_sent = ['these', 'are', 'cats']
    inv_cat_sent = ['these', 'are', '<unk>']
    assert vocab.to_tokens(vocab.to_ids(cat_sent)) == inv_cat_sent


def test_vocab_save_load():
    voc_1 = copy.deepcopy(vocab)
    voc_2 = wr.Vocab(unk_token='<u>')

    voc_1.save('test/test_vocab.yaml')
    assert voc_1.tok_to_id == vocab.tok_to_id
    assert voc_1.id_to_tok == vocab.id_to_tok

    voc_1.load('test/test_vocab.yaml')
    voc_2.load('test/test_vocab.yaml')
    assert voc_1.tok_to_id == voc_2.tok_to_id == vocab.tok_to_id
    assert voc_1.id_to_tok == voc_2.id_to_tok == vocab.id_to_tok

    os.remove('test/test_vocab.yaml')

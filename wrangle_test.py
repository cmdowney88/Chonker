import pytest
from chonker import wrangle as wr

lines = wr.get_lines('test_lines.txt')

def test_get_lines():
    assert lines[0] == 'thisisastringofcharacters'
    assert lines[1] == 'these words  are separated by spaces'
    assert lines[2] == 'these\twords\tare\ttab\tdelimited'


ws_split_lines = wr.split_lines(lines)
space_split_lines = wr.split_lines(lines, delimiter=r'\ +')
tab_split_lines = wr.split_lines(lines, delimiter=r'\t+')

def test_ws_split_lines():
    assert ws_split_lines[0] == ['thisisastringofcharacters']
    assert ws_split_lines[1] == ['these', 'words', 'are', 'separated', 'by', 'spaces']
    assert ws_split_lines[2] == ['these', 'words', 'are', 'tab', 'delimited']

def test_space_split_lines():
    assert space_split_lines[0] == ['thisisastringofcharacters']
    assert space_split_lines[1] == ['these', 'words', 'are', 'separated', 'by', 'spaces']
    assert space_split_lines[2] == ['these\twords\tare\ttab\tdelimited']

def test_tab_split_lines():
    assert tab_split_lines[0] == ['thisisastringofcharacters']
    assert tab_split_lines[1] == ['these words  are separated by spaces']
    assert tab_split_lines[2] == ['these', 'words', 'are', 'tab', 'delimited']


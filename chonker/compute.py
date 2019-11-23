import numpy as np

def acyclic_viterbi(transitions: np.ndarray):
    shape = transitions.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    seq_length = shape[0]

    trellis = {}
    trellis[0] = (0.0, 'bos')
    for position in range(1,seq_length+1):
        trellis[position] = None
        candidates = transitions[:,[position-1]].T[0]
        for previous in range(position):
            candidates[previous] += trellis[previous][0]
        trellis[position] = (np.max(candidates), np.argmax(candidates))

    position = seq_length
    previous = trellis[position][1]
    best_prob = trellis[position][0]
    best_path = [(previous,position)]
    while previous > 0:
        position = previous
        previous = trellis[position][1]
        best_path.append((previous,position))

    best_path.reverse()
    return best_path, best_prob

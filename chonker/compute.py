import numpy as np
import sys


def acyclic_viterbi(transitions: np.ndarray, mode='max', length=None):
    if mode != 'max' and mode != 'min':
        sys.exit(
            f'Mode {best} is not recognized. Valid modes are `max` and `min`'
        )

    shape = transitions.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]

    if length:
        assert length <= shape[0]
        seq_length = length
    else:
        seq_length = shape[0]

    trellis = {}
    trellis[0] = (0.0, -1)
    for position in range(1, seq_length + 1):
        trellis[position] = None
        candidates = transitions[:, [position - 1]].T[0]
        for previous in range(position):
            candidates[previous] += trellis[previous][0]
        if mode == 'max':
            trellis[position] = (np.max(candidates), np.argmax(candidates))
        else:
            trellis[position] = (np.min(candidates), np.argmin(candidates))

    position = seq_length
    previous = trellis[position][1]
    best_prob = trellis[position][0]
    best_path = [(previous, position)]
    while previous > 0:
        position = previous
        previous = trellis[position][1]
        best_path.append((previous, position))

    best_path.reverse()
    return best_path, best_prob

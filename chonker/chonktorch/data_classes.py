import torch
import torch.nn as nn
from torch.utils.data import Dataset
from . import functions as ctf


class VariableLengthDataset(Dataset):
    """
    A PyTorch Dataset subclass for padded datasets of variable-length sequences
    in which it is undesirable to break the sequences into "streams" of a single
    length

    The Dataset is initialized into sorted batches of sequences, padded to the
    max sequence length of the batch. The batch size can be defined either as
    a set number of sequences, or as an approximate number of tokens/timesteps
    per batch.

    Args:
        data: A list of numerically-embedded sequences
        batch_size: Batch size, either by number of sequences or number of
            tokens/timesteps. See ``batch_by``
        pad_value: The value with which to pad sequences within a batch
            that are shorter than the max batch length
        batch_by (optional): Unit by which to batch. If ``sequences``, a batch
            will consist of number ``batch_size`` sequences. If ``tokens``, a
            batch will consist of sequences that add up to no more than
            ``batch_size`` tokens total when padded. Note under this option,
            batches will contain different number of sequences, but be about
            the same size in terms of total number of elements. Default:
            ``sequences``
        gradient_accumulation (optional): number of gradient accumulation steps
            over batches. Used to define the effective size of full batches when
            dropping incomplete the incomplete final batch. See ``drop_final``.
            Default: 1
        drop_final (optional): Whether to drop all final sequences that do not
            add up to a full batch (with gradient accumulation). Default: 
            ``True``
        max_padding (optional): The maximum number of padding tokens allowed at
            the end of any sequence (padding and attention masks can interact
            in ways that throw errors in PyTorch transformer implementations).
            Default: ``None``
        max_pad_strategy (optional): The strategy to deal with batches that
            require more padding than the maximum. If ``split``, no sequences
            will be dropped from the dataset, and the batch will simply be split
            at the last point at which the number of pad tokens is legal. NOTE:
            this may lead to more inconsistent batch sizes. If ``soft_drop``,
            the batch will be dropped up until the length differential that
            causes the large number of pad tokens, and batching will continue
            from that index. If ``hard_drop``, the entire candidate batch will
            be dropped. This may be useful if batching is being done by
            sequences and only complete batch sizes are acceptable (i.e.
            ``drop_final`` is ``True``). Default: ``soft_drop``
    """
    def __init__(
        self,
        data: list,
        batch_size: int,
        pad_value: float,
        batch_by: str = 'sequences',
        gradient_accumulation: int = 1,
        drop_final: bool = False,
        max_padding: int = None,
        max_pad_strategy: str = 'soft_drop'
    ) -> Dataset:

        self.batch_size = batch_size
        self.pad_value = pad_value
        self.max_padding = max_padding
        self.max_pad_strategy = max_pad_strategy

        # Make sure that all string parameters have recognized values
        if batch_by not in ['sequences', 'tokens']:
            raise ValueError(f'Batch-by option {self.batch_by} is not valid')
        if max_pad_strategy not in ['split', 'soft_drop', 'hard_drop']:
            raise ValueError(
                f'Max pad strategy {self.max_pad_strategy} is not valid'
            )

        # If batches are formed by a set number of sequences and drop_final is
        # True, calculate the total number of full batches and drop leftover
        # sequences
        if batch_by == 'sequences' and drop_final:
            full_batch_size = batch_size * gradient_accumulation
            num_full_batches = len(data) // full_batch_size
            data = data[:num_full_batches * full_batch_size]

        # Sort the data sequences in descending order of length for efficiency
        # (save permutations used to sort and unsort sequences as object
        # attributes)
        lengths = [len(line) for line in data]
        self.sort_pmt, self.unsort_pmt = ctf.sort_unsort_pmts(
            lengths, descending=True
        )
        data = list(zip(data, lengths))
        data = [data[i] for i in self.sort_pmt]

        # Form the sorted data into batches, either by number of sequences or
        # approximate number of tokens/timesteps. Collate all batches into
        # tensors using the pad_collate function
        self.total_num_instances = len(data)
        batched_data = []
        current_index = 0
        while current_index < self.total_num_instances:
            if batch_by == 'sequences':
                next_index = current_index + batch_size
            elif batch_by == 'tokens':
                current_length = data[current_index][1]
                seqs_per_batch = max(batch_size // current_length, 1)
                next_index = current_index + seqs_per_batch

            next_index = min(self.total_num_instances, next_index)
            
            # Some combinations of padding and attention patterns in pytorch
            # cause an error during forward passes, so it is sometimes necessary
            # to restrict the number of pad tokens allowed at the end of a
            # sequence
            if self.max_padding:
                legal_next_index = current_index + 1
                current_length = data[current_index][1]
                for i in range(current_index + 1, next_index):
                    length_difference = (
                        current_length - data[i][1] 
                    )
                    if length_difference > max_padding:
                        break
                    else:
                        legal_next_index = i + 1
                if (legal_next_index < next_index):
                    if self.max_pad_strategy == 'hard_drop':
                        current_index = next_index
                        continue
                    elif self.max_pad_strategy == 'soft_drop':
                        current_index = legal_next_index
                        continue
                next_index = legal_next_index

            new_batch = self.pad_collate(data[current_index:next_index])
            batched_data.append(new_batch)
            current_index = next_index

        # If batches are formed by number of tokens and drop_final is True,
        # calculate the total number of full batches and drop leftover non-full
        # batches
        if batch_by == 'tokens' and drop_final:
            num_full_batches = len(batched_data) // gradient_accumulation
            num_batches = num_full_batches * gradient_accumulation
            trimmed_instances = sum(
                [len(batch[1]) for batch in batched_data[num_batches:]]
            )
            batched_data = batched_data[:num_batches]
            self.total_num_instances -= trimmed_instances

        # Data is a list of collated batches (tensor, lengths)
        self.data = batched_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def pad_collate(self, batch):
        """
        Collate and pad a list of (sequence, length) pairs into a contiguous
        tensor and list of lengths
        """
        sequences = [torch.tensor(x[0]) for x in batch]
        lengths = [x[1] for x in batch]
        tensor = nn.utils.rnn.pad_sequence(
            sequences, padding_value=self.pad_value
        )
        tensor = tensor.contiguous()
        return (tensor, lengths)

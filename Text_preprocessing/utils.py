import collections
from d2l import torch as d2l
import random
import torch



def tokenize(lines, token='word'):
    """split text to words and chars"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('error：unknown char：' + token)

class Vocab:
    """words dict"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # sort with frequence
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # unknown char's index is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0 # unknown char's index is 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """sum up chars frequency"""
    # tokens is 1D or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # flattern chars list to one flat list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def seq_data_iter_random(corpus, batch_size, num_steps):
    """use random selection to generate a small batch sequence"""
    # Partition the sequence starting from a random offset, with the random range including num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # The subtraction of 1 is because we need to account for the label
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting index of a subsequence of length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # During the iteration of random sampling，
    # the subsequences from two adjacent random mini-batches are not necessarily adjacent in the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length num_steps starting from position pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, initial_indices contains the random starting indices of the subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a mini-batch of subsequences using sequential partitioning"""
    # Partition the sequence starting from a random offset
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """The iterator for loading sequence data"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_text(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """return The dataset iterator and the vocabulary."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
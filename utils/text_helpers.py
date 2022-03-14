## This Lab is made with inspiration & code snippets of:
##     - https://github.com/yxtay/char-rnn-text-generation
##     - https://d2l.ai/chapter_recurrent-modern/lstm.html
##     - https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html

import numpy as np
import random
import string
import collections
import re


def count_corpus(tokens):  # @save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(lines, token='word'):  # @save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def read_file(input_file):  # @save
    """Load the time machine dataset into a list of text lines."""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus(input_file, max_tokens=-1):  # @save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_file(input_file)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [token for line in tokens for token in line]
    # when need to get decoded corpus:
    #corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class Vocab:  # @save
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        # The index for the unknown token is 0
        # TODO: fix here for Uknown cases
        self._idx_to_token = ['<unk>'] + reserved_tokens
        self._token_to_idx = {token: idx
                              for idx, token in enumerate(self._idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self._token_to_idx:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self._idx_to_token[indices]
        return [self._idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


###
# data processing
###


def encode_text(text, char2id):
    """
    encode text to array of integers with CHAR2ID
    """
    return np.fromiter((char2id[ch] for ch in text), int)


def decode_text(int_array, id2char):
    """
    decode array of integers to text with ID2CHAR
    """
    return "".join((id2char[ch] for ch in int_array))


def one_hot_encode(indices, num_classes):
    """
    one-hot encoding
    """
    return np.eye(num_classes)[indices]


def batch_generator(sequence, batch_size=64, seq_len=64, one_hot_features=False, one_hot_labels=False, vocab=Vocab):
    """
    batch generator for sequence
    ensures that batches generated are continuous along axis 1
    so that hidden states can be kept across batches and epochs
    """

    # calculate effective length of text to use
    num_batches = (len(sequence) - 1) // (batch_size * seq_len)
    if num_batches == 0:
        raise ValueError("No batches created. Use smaller batch size or sequence length.")
    print("number of batches: %s." % num_batches)
    rounded_len = num_batches * batch_size * seq_len
    print("effective text length: %s." % rounded_len)

    VOCAB_SIZE = len(vocab)

    x = np.reshape(sequence[: rounded_len], [batch_size, num_batches * seq_len])
    if one_hot_features:
        x = one_hot_encode(x, VOCAB_SIZE)
    print("x shape: ", x.shape)

    y = np.reshape(sequence[1: rounded_len + 1], [batch_size, num_batches * seq_len])
    if one_hot_labels:
        y = one_hot_encode(y, VOCAB_SIZE)
    print("y shape: ", y.shape)

    epoch = 0
    while True:
        # roll so that no need to reset rnn states over epochs
        x_epoch = np.split(np.roll(x, -epoch, axis=0), num_batches, axis=1)
        y_epoch = np.split(np.roll(y, -epoch, axis=0), num_batches, axis=1)
        for batch in range(num_batches):
            yield x_epoch[batch], y_epoch[batch]
        epoch += 1


###
# text generation
###

def generate_seed(text, seq_lens=(2, 4, 8, 16, 32)):
    """
    select subsequence randomly from input text
    """
    # randomly choose sequence length
    seq_len = random.choice(seq_lens)
    # randomly choose start index
    start_index = random.randint(0, len(text) - seq_len - 1)
    seed = text[start_index: start_index + seq_len]
    return seed


def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    # need 64 floating point precision
    probs = np.array(probs, dtype=np.float64)
    # set probabilities after top_n to 0
    probs[np.argsort(probs)[:-top_n]] = 0
    # renormalise probabilities
    probs /= np.sum(probs)
    sampled_index = np.random.choice(len(probs), p=probs)
    return sampled_index


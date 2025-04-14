import torchaudio.datasets as datasets
import pickle as pkl
from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def tokenize(text, merges):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

if __name__ == "__main__":
    data = datasets.LIBRISPEECH("../../data", url="dev-clean", )
    corpus = []
    for i in range(2800):
        try:
            corpus.append(data.__getitem__(i)[2])

        except IndexError as err:
            break

    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []

    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()

    vocab = ["<|startoftext|>", "<|endoftext|>", "<|padding|>"] + alphabet.copy()
    merges = dict()
    splits = {word: [c for c in word] for word in word_freqs.keys()}

    vocab_size = 128

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    print(vocab)
    with open("vocabulary.pkl", "wb") as f:
        f.write(pkl.dumps(vocab))

    with open("merges.pkl", "wb") as f:
        f.write(pkl.dumps(merges))
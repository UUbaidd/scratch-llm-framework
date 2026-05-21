# custom_dl/tokenizers/bpe.py
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def fit(self, raw_vocab, merges=5):
        self.vocab = raw_vocab
        for _ in range(merges):
            p = self.get_stats(self.vocab)
            if not p: break
            best = max(p, key=p.get)
            self.vocab = self.merge_vocab(best, self.vocab)

        all_tokens = sorted(list(set(t for w in self.vocab.keys() for t in w.split())))
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, text):
        """
        Converts a space-separated string of tokens/characters into a list of token IDs.
        Safely ignores or warns about characters/tokens not present in the vocabulary.
        """
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                print(f"Warning: Token '{token}' not found in vocabulary distribution configuration.")
        return ids

    def decode(self, ids):
        """
        Converts a list of token IDs back into a single space-separated string.
        """
        return ' '.join([self.id_to_token[i] for i in ids if i in self.id_to_token])
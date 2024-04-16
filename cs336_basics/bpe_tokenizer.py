from cs336_basics.common_config import RESULT_PATH, DATA_PATH 
from .utils import dict_adding
from typing import IO, BinaryIO, Iterable, Optional, Type, Iterator
import json
import regex
import pickle
import gc 
from tests.common import gpt2_bytes_to_unicode

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = [tuple(merge) for merge in merges]
        self.special_tokens = sorted(special_tokens, key=lambda token: len(token), reverse=True) if special_tokens else []
        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab.values():
                new_id = len(self.vocab)
                self.vocab[new_id] = token.encode('utf-8')
                self.reverse_vocab[token.encode('utf-8')] = new_id
        # Make merges into a dictionary for faster lookup
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.encode_cache = {}
        
    def encode(self, text: str) -> list[int]:
        """
        Given a string, return a list of integers representing the byte-pair encoding of the text.
        """
        # Pre-tokenize the text
        pre_token_lst = self._pre_tokenize(text)
        encoded_text = []
        for pre_token in pre_token_lst:
            encoded_pre_token_int_lst = self._merge_tokens(pre_token)
            encoded_text.extend(encoded_pre_token_int_lst)
        return encoded_text

    
    def decode(self, tokens: list[int]) -> str:
        """
        Given a list of integers, return the decoded string.
        """
        total = b''
        for token in tokens:
            total += self.vocab[token]
        return total.decode('utf-8', errors='replace')
        
        
    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Pre-tokenize the text by replacing special tokens with a unique token ID.
        """
        # TODO: Reorder the special tokens 
        parts = [text]
        if self.special_tokens:
            special_pattern = '|'.join(regex.escape(token) for token in self.special_tokens)
            parts = regex.split(f'({special_pattern})', text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_token_lst = []
        for part in parts:
            if part in self.special_tokens:
                pre_token_lst.append(part)
            else:
                subparts = regex.findall(PAT, part)
                pre_token_lst.extend(subparts) 

        return pre_token_lst

    def _merge_tokens(self, pre_token: str) -> list[int]:
        if pre_token in self.special_tokens:
            return [self.reverse_vocab[pre_token.encode('utf-8')]]
        if pre_token in self.encode_cache:
            return self.encode_cache[pre_token]
        
        encoded_pre_token_int_lst = pre_token.encode('utf-8')
        token_lst = [bytes([a]) for a in encoded_pre_token_int_lst]
        while len(token_lst) > 1:
            pair_lst = [(p1, p2) for p1, p2 in zip(token_lst[:-1], token_lst[1:])]
            min_pair = min(pair_lst, key=lambda x: self.merges_dict.get(x, float('inf')))
            if min_pair not in self.merges_dict:
                break
            # Loop over the token list to merge all occurrences of the pair
            index = 0
            while index < len(token_lst) - 1:
                if token_lst[index] == min_pair[0] and token_lst[index + 1] == min_pair[1]:
                    token_lst[index] = min_pair[0] + min_pair[1]
                    token_lst.pop(index + 1)
                index += 1    
                    
        self.encode_cache[pre_token] = [self.reverse_vocab[token] for token in token_lst]
        
        return self.encode_cache[pre_token]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required
        for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            encoded_line = self.encode(chunk)
            for token in encoded_line:
                yield token
                
       
if __name__ == "__main__":
    with open(RESULT_PATH / "test_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open(RESULT_PATH / "test_merges.txt", "r") as f:
        merges = [tuple(line.rstrip().split(" ")) for line in f]
    tokenizer = BPETokenizer(vocab, merges, [])
    text = "🙃"
    encoded_text = tokenizer.encode(text)
    print(encoded_text)
    decoded_text = tokenizer.decode(encoded_text)
    print(decoded_text)
from cs336_basics.common_config import RESULT_PATH, DATA_PATH 
import json
from time import time
from memory_profiler import memory_usage
import pickle
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from cs336_basics.utils import dict_adding, dict_dict_add_key, dict_dict_del_key
import regex as re
from .bpe_tokenizer import BPETokenizer
import gc

def merged_pre_token(pre_token, merging_pair, pair_freq, pre_tokenized_freq, pre_token_to_id, pair_to_pre_token_id):
    token_list = list(pre_token)
    pre_token_id = pre_token_to_id[pre_token]
    index = 0
    
    while index < len(token_list) - 1:
        if (token_list[index], token_list[index + 1]) == merging_pair:
            new_token = token_list[index] + token_list[index+1]
            if index > 0:
                dict_adding(pair_freq, (token_list[index - 1], token_list[index]), -pre_tokenized_freq)
                dict_adding(pair_freq, (token_list[index - 1], new_token), pre_tokenized_freq)
               
                dict_dict_del_key(pair_to_pre_token_id, (token_list[index - 1], token_list[index]), pre_token_id)
                dict_dict_add_key(pair_to_pre_token_id, (token_list[index - 1], new_token), pre_token_id)
            if index < len(token_list) - 2:
                dict_adding(pair_freq, (token_list[index + 1], token_list[index + 2]), -pre_tokenized_freq)
                dict_adding(pair_freq, (new_token, token_list[index + 2]), pre_tokenized_freq)
                dict_dict_del_key(pair_to_pre_token_id, (token_list[index + 1], token_list[index + 2]), pre_token_id)
                dict_dict_add_key(pair_to_pre_token_id, (new_token, token_list[index + 2]), pre_token_id)
            
            token_list = token_list[:index] + [new_token] + token_list[index + 2:]
        index += 1

    return tuple(token_list)

def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # # Load Data
    # with open(input_path, "r") as f:
    #     data = f.read()
        
    # Initialize vocab
    # Include all bytes in the vocab    
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    merges = []
    
    # Pre-tokenize data
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_size=1024 * 1024 * 1024 # 1GB
    pre_tokenized_dict = {}
    pre_token_list = []
    pre_token_to_id = {}
    
    with open(input_path, "r", encoding='utf-8') as file:
        buffer = ""
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                if buffer:
                    pre_tokenize_buffer(buffer, pre_tokenized_dict, pre_token_list, pre_token_to_id, PAT)
                break
            buffer += chunk
            last_newline = buffer.rfind('\n')
            if last_newline != -1:
                process_text = buffer[:last_newline]  # process up to the last complete line
                buffer = buffer[last_newline + 1:]  # keep the rest
                pre_tokenize_buffer(process_text, pre_tokenized_dict, pre_token_list, pre_token_to_id, PAT)
        
    gc.collect()
    
    # pre_tokenized = re.findall(PAT, data)
    # pre_tokenized_dict = {}
    # pre_token_list = []
    # pre_token_to_id = {}
    
    # for pre_token in pre_tokenized:
    #     encoded_pre_token_int_lst = list(pre_token.encode('utf-8'))
    #     encoded_pre_token = tuple([bytes([a]) for a in encoded_pre_token_int_lst])
    #     if encoded_pre_token not in pre_token_to_id:
    #         pre_tokenized_dict[len(pre_token_list)] = 1
    #         pre_token_to_id[encoded_pre_token] = len(pre_token_list)
    #         pre_token_list.append(encoded_pre_token)
    #     else:
    #         pre_tokenized_dict[pre_token_to_id[encoded_pre_token]] += 1
    # del pre_tokenized
    # gc.collect() 
        
    # Initialize pair frequency
    pair_freq = {}
    pair_to_pre_token_id = {}
    for pre_token in pre_token_list:
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            if pair not in pair_freq:
                pair_freq[pair] = pre_tokenized_dict[pre_token_to_id[pre_token]]
                pair_to_pre_token_id[pair] = {pre_token_to_id[pre_token]: 1}
            else:
                pair_freq[pair] += pre_tokenized_dict[pre_token_to_id[pre_token]]
                dict_adding(pair_to_pre_token_id[pair], pre_token_to_id[pre_token], 1)
                

    # Train BPE
    while len(vocab) + len(special_tokens) < vocab_size:
        max_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
        new_token = max_pair[0] + max_pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token 
        merges.append(max_pair)
        
        for pre_token_id in pair_to_pre_token_id[max_pair]:
            pre_token = pre_token_list[pre_token_id]
            new_pre_token = merged_pre_token(pre_token, merging_pair=max_pair, pair_freq=pair_freq, pre_tokenized_freq=pre_tokenized_dict[pre_token_id], pre_token_to_id=pre_token_to_id, pair_to_pre_token_id=pair_to_pre_token_id)
            if new_pre_token != pre_token:
                pre_token_list[pre_token_id] = new_pre_token
                pre_token_to_id[new_pre_token] = pre_token_id
                del pre_token_to_id[pre_token]
                        
        del pair_to_pre_token_id[max_pair]
        del pair_freq[max_pair]
        
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
        
    return vocab, merges
        
def pre_tokenize_buffer(buffer, pre_tokenized_dict, pre_token_list, pre_token_to_id, PAT):
    pre_tokenized = re.findall(PAT, buffer)

    for pre_token in pre_tokenized:
        encoded_pre_token_int_lst = list(pre_token.encode('utf-8'))
        encoded_pre_token = tuple([bytes([a]) for a in encoded_pre_token_int_lst])
        if encoded_pre_token not in pre_token_to_id:
            pre_tokenized_dict[len(pre_token_list)] = 1
            pre_token_to_id[encoded_pre_token] = len(pre_token_list)
            pre_token_list.append(encoded_pre_token)
        else:
            pre_tokenized_dict[pre_token_to_id[encoded_pre_token]] += 1
    

def save_vocab(vocab, path):
    # Convert the byte values in the vocab back to their gpt2 string representation.
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    converted_vocab = {k: ''.join(gpt2_byte_encoder[b] for b in v) for k, v in vocab.items()}
    with open(path, 'w') as f:
        json.dump(converted_vocab, f)

def save_merges(merges, path):
    # Convert the byte tuples back to their gpt2 string representation.
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    converted_merges = [
        (''.join(gpt2_byte_encoder[b] for b in m1), ''.join(gpt2_byte_encoder[b] for b in m2))
        for m1, m2 in merges
    ]
    with open(path, 'w') as f:
        for merge_pair in converted_merges:
            f.write(' '.join(merge_pair) + '\n')

def load_vocab(vocab_path):
    """
    Load the vocabulary from a JSON file.

    :param vocab_path: Path to the vocabulary file.
    :return: Dictionary of vocabulary items.
    """
    # Compare the vocab to the expected output vocab
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as f:
        vocab_ = json.load(f)
        vocab = {
            int(gpt2_vocab_index): bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_index, gpt2_vocab_item in vocab_.items()
        }
    return vocab

def load_merges(merges_path):
    """
    Load the merges from a text file.

    :param merges_path: Path to the merges file.
    :return: List of tuple pairs representing merges.
    """
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(merges_path) as f:
        merges_ = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in merges_
        ]
    return merges

def load_tokenizer(vocab_path, merges_path, special_tokens):
    vocab = load_vocab(vocab_path)
    merges = load_merges(merges_path)
    return BPETokenizer(vocab, merges, special_tokens)

def train_bpe_from_dataset(dataset_name, vocab_size, special_tokens):
    start_time = time()  # Start timing
    initial_mem = memory_usage()[0]  # Initial memory usage
    
    if dataset_name == "owt2":
        input_path = f"{DATA_PATH}/owt_train.txt"
    elif dataset_name == "tinystories":
        input_path = f"{DATA_PATH}/TinyStoriesV2-GPT4-train.txt"
    elif dataset_name == "test":
        input_path = f"/sailhome/jiangm/spring2024-assignment1-basics/tests/fixtures/corpus.en"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens)
        
    save_merges(merges, RESULT_PATH / f"{dataset_name}_merges.txt")
    save_vocab(vocab, RESULT_PATH / f"{dataset_name}_vocab.json")
            
    end_time = time()  # End timing
    final_mem = memory_usage()[0]  # Final memory usage

    # Calculate time and memory usage
    total_time = end_time - start_time
    total_memory = final_mem - initial_mem

    return total_time, total_memory

if __name__ == "__main__":
    # dataset = "owt2"
    # total_time, total_memory = train_bpe_from_dataset(dataset, 32000, ['<|endoftext|>'])
    # # total_time, total_memory = train_bpe_from_dataset(dataset, 500, [])
    # print(f"Time taken: {total_time/3600} hours")
    # print(f"Memory used: {total_memory /1024} GB")
    
    merges = load_merges(RESULT_PATH / f"test_merges.txt")
    print(merges)
    vocab = load_vocab(RESULT_PATH / f"test_vocab.json")
    print(vocab)
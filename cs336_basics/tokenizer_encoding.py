import numpy as np
from .common_config import DATA_PATH, RESULT_PATH
from .bpe_tokenizer import BPETokenizer
from .train_bpe import load_tokenizer
import json
import regex
from concurrent.futures import ProcessPoolExecutor
from typing import List
import difflib


"""Using your TinyStories and OpenWebText tokenizers, encode the respective training and development
datasets into a sequence of integer token IDs. We’ll use this later to train our language
model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
uint16 an appropriate choice?"""


def process_text(text):
    """Process the text to encode it."""
    encoded_text = tokenizer.encode(text)
    return np.array(encoded_text, dtype=np.uint16)

def save_encoded_texts(encoded_texts, save_path):
    """Save all encoded texts to a file in order."""
    mode = 'wb'  # write binary mode
    for data in encoded_texts:
        with open(save_path, mode) as f:
            np.save(f, data, allow_pickle=False)
        mode = 'ab'  # append binary mode

def encode_and_save_dataset(tokenizer, dataset_name):
    """Process large text files in chunks and save encoded texts incrementally."""
    if dataset_name == "owt2_train":
        input_path = f"{DATA_PATH}/owt_train.txt"
    elif dataset_name == "owt2_valid":
        input_path = f"{DATA_PATH}/owt_valid.txt"
    elif dataset_name == "tinystories_train":
        input_path = f"{DATA_PATH}/TinyStoriesV2-GPT4-train.txt"
    elif dataset_name == "tinystories_valid":
        input_path = f"{DATA_PATH}/TinyStoriesV2-GPT4-valid.txt"
    elif dataset_name == "test":
        input_path = f"/sailhome/jiangm/spring2024-assignment1-basics/tests/fixtures/corpus.en"
        
    output_path = f"{DATA_PATH}/{dataset_name}.npy"
    
    chunk_size = 1024 * 10 # 10MB

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        with open(input_path, "r", encoding='utf-8') as file:
            buffer = ""
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    if buffer:  # process remaining buffer if no new lines and EOF is reached
                        futures.append(executor.submit(process_text, buffer))
                    break
                buffer += chunk
                last_newline = buffer.rfind('\n')
                if last_newline != -1:
                    process_text_chunk = buffer[:last_newline]  # process up to the last complete line
                    buffer = buffer[last_newline + 1:]  # keep the rest
                    futures.append(executor.submit(process_text, process_text_chunk))
                elif len(buffer) > chunk_size:  # If buffer exceeds chunk size and no newline, process full buffer
                    futures.append(executor.submit(process_text, buffer))
                    buffer = ""  # Reset buffer
        
        results = [future.result() for future in futures]
    
    save_encoded_texts(results, output_path)


if __name__ == "__main__":
    dataset = 'owt2_valid'
    tokenizer = load_tokenizer(vocab_path=RESULT_PATH / f"owt2_vocab.json", merges_path=RESULT_PATH / f"owt2_merges.txt", special_tokens=[])
    encode_and_save_dataset(tokenizer, dataset)
    
    # dataset = 'test'
    # tokenizer = load_tokenizer(vocab_path=RESULT_PATH / f"test_vocab.json", merges_path=RESULT_PATH / f"test_merges.txt", special_tokens=[])
    
    # with open(f"/sailhome/jiangm/spring2024-assignment1-basics/tests/fixtures/corpus.en", "r", encoding='utf-8') as f:
    #     text = f.read()

    
    
    # encode_and_save_dataset(tokenizer, "test")

    # # Load the encoded dataset
    # encoded_dataset = np.load(f"{DATA_PATH}/test.npy")
    # reference_text = tokenizer.decode(encoded_dataset)
    
    # find_diffs = lambda s1, s2: '\n'.join(difflib.ndiff(s1.split(), s2.split()))
    # print(find_diffs(text, reference_text))
    
    

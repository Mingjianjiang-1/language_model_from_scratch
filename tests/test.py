from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, test_encode_iterable_tinystories_sample_roundtrip
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from .adapters import get_tokenizer
import tiktoken


VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
corpus_path = FIXTURES_PATH / "address.txt"
with open(corpus_path) as f:
	text = f.read()

print('reference')
tokenizer_ref = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer_ref.encode(text)
print(encoded_text)
decoded_text = tokenizer_ref.decode(encoded_text)
print(decoded_text)


tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"])
print('test')
encoded_text = tokenizer.encode(text)
print(encoded_text)
decoded_text = tokenizer.decode(encoded_text)
print(decoded_text)


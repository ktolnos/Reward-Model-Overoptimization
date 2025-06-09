from typing import List
from transformers import PreTrainedTokenizerBase, BatchEncoding


class TokenizerWrapper(PreTrainedTokenizerBase):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token

    def __call__(self, *args, **kwargs):
        print("call")
        text = kwargs['text']
        pattern = "<end_of_turn>\n<start_of_turn>model\n"
        if pattern not in text:
            if isinstance(text, str):
                kwargs['text'] = text + pattern
            elif isinstance(text, list):
                kwargs['text'] = [t + pattern for t in text]
            else:
                raise ValueError(f"Unsupported type for text: {type(text)}")

        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs) -> List[int]:
        print("encode")
        return tokenizer.encode(*args, **kwargs)

    def encode_plus(self, *args, **kwargs) -> BatchEncoding:
        print("encode_plus")
        return tokenizer.encode_plus(*args, **kwargs)

    def batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        print("batch_encode_plus")
        return tokenizer.batch_encode_plus(*args, **kwargs)

    def tokenize(self, *args, **kwargs) -> List[str]:
        print("tokenize")
        return tokenizer.tokenize(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        print("decode")
        return tokenizer.decode(*args, **kwargs)
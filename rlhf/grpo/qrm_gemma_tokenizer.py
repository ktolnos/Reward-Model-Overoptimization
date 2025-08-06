from typing import List
from transformers import PreTrainedTokenizerBase, BatchEncoding
import transformers
transformers.models.llama.modeling_llama.LLAMA_INPUTS_DOCSTRING = "No docstring."


class TokenizerWrapper(PreTrainedTokenizerBase):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_path, **kwargs):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.model_path = model_path

        pattern = "<end_of_turn>\n<start_of_turn>model\n" if 'gemma' in self.model_path else """<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        tokenized_pattern = tokenizer(pattern, add_special_tokens=False, return_tensors='pt')
        self.pattern = pattern
        self.tokenized_pattern = tokenized_pattern.input_ids[0]



    def __call__(self, *args, **kwargs):
        text = kwargs.get('text', None)
        if text is None:
            text = args[0]
            args = args[1:]
        if isinstance(text, str):
            if pattern not in text:
                kwargs['text'] = text + pattern
        elif isinstance(text, list):
            for i, s in enumerate(text):
                if pattern not in s:
                    text[i] = s + pattern
            kwargs['text'] = text
        else:
            raise ValueError(f"Unsupported type for text: {type(text)}")
        result = self.tokenizer(*args, **kwargs)
        if 'input_ids' in result:
            result['input_ids'][:, -len(self.tokenized_pattern):] = self.tokenized_pattern
        if 'attention_mask' in result:
            result['attention_mask'][:, -len(self.tokenized_pattern):] = 1
        if 'token_type_ids' in result:
            result['token_type_ids'][:, -len(self.tokenized_pattern):] = 0
        return result

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs) -> List[int]:
        print("encode")
        return self.tokenizer.encode(*args, **kwargs)

    def encode_plus(self, *args, **kwargs) -> BatchEncoding:
        print("encode_plus")
        return self.tokenizer.encode_plus(*args, **kwargs)

    def batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        print("batch_encode_plus")
        return self.tokenizer.batch_encode_plus(*args, **kwargs)

    def tokenize(self, *args, **kwargs) -> List[str]:
        print("tokenize")
        return self.tokenizer.tokenize(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        print("decode")
        return self.tokenizer.decode(*args, **kwargs)
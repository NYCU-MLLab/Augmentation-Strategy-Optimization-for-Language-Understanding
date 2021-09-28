import torch
import tensorflow_hub as hub

from typing import Dict
from sentence_transformers import SentenceTransformer
# from .utils import get_sentence_from_text_field_tensors
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


def get_sentence_from_text_field_tensors(
    transformer_tokenizer,
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
):
    sentences_token_ids = text_field_tensors["tokens"]["token_ids"].int().tolist()
    return transformer_tokenizer.tokenizer.decode(
        sentences_token_ids[0],
        skip_special_tokens=True
    )


class TextEmbedder(BasicTextFieldEmbedder):
    def __init__(self,
                 token_embedders: Dict):
        super(TextEmbedder, self).__init__(token_embedders)


class SentenceEmbedder(object):
    def __init__(
        self,
        tokenizer,
        model_url: str = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    ):
        super(SentenceEmbedder, self).__init__()
        self.tokenizer = tokenizer
        self.model = SentenceTransformer(model_url, device="cpu")

    def __call__(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        input_str = get_sentence_from_text_field_tensors(
            self.tokenizer,
            state
        )

        return self.model.encode([input_str], convert_to_tensor=True)


class UniversalSentenceEmbedder(object):
    def __init__(
        self,
        tokenizer,
        model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ):
        super(UniversalSentenceEmbedder, self).__init__()
        self.tokenizer = tokenizer
        self.model = hub.load(model_url)

    def __call__(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        input_str = get_sentence_from_text_field_tensors(
            self.tokenizer,
            state
        )

        return self.model([input_str])


def main():
    pass


if __name__ == '__main__':
    main()

import sacremoses as sm

from typing import List
from allennlp.data.vocabulary import Vocabulary


class WordTokenizer(object):
    def __init__(
        self,
        vocab=None,
        language="en"
    ):
        super(WordTokenizer, self).__init__()
        self._tokenizer = sm.MosesTokenizer(
            language
        )
        self._detokenizer = sm.MosesDetokenizer(
            language
        )
        self._vocab = vocab

    def tokenize(
        self,
        input_str: str
    ):
        return self._tokenizer.tokenize(input_str)

    def covert_tokens_to_ids(
        self,
        input_tokens: List[str]
    ):
        input_ids = []

        for input_token in input_tokens:
            input_id = self._vocab.get_token_index(input_token)
            input_ids.append(input_id)

        return input_ids

    def encode(
        self,
        input_str: str
    ):
        assert (self._vocab), "Vocabulary have not decided yet!"

        input_tokens = self.tokenize(input_str)

        return self.covert_tokens_to_ids(input_tokens)

    def detokenize(
        self,
        input_tokens: List[str]
    ):
        return self._detokenizer.detokenize(input_tokens)

    def convert_ids_to_tokens(
        self,
        input_ids: List[int]
    ):
        input_tokens = []

        for input_id in input_ids:
            input_token = self._vocab.get_token_from_index(input_id)
            input_tokens.append(input_token)

        return input_tokens

    def decode(
        self,
        input_ids: List[int]
    ):
        assert (self._vocab), "Vocabulary have not decided yet!"

        input_tokens = self.convert_ids_to_tokens(
            input_ids
        )

        return self.detokenize(input_tokens)

    def index_with(
        self,
        vocab: Vocabulary
    ):
        self._vocab = vocab


def main():
    pass


if __name__ == '__main__':
    main()

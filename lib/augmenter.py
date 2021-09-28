import abc
import copy
import math
import torch
import random

from .utils import pad_text_tensor_list, unpad_text_field_tensors
from .tokenizer import WordTokenizer
from typing import Dict, List
from overrides import overrides
from nltk.corpus import wordnet
from allennlp.nn.util import move_to_device
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Augmenter(object):
    def __init__(
        self,
        dataset_dict: Dict,
        tokenizer=None
    ):
        self.detokenizer = dataset_dict["dataset_reader"].transformer_tokenizer
        self.transformer_vocab = dataset_dict["dataset_reader"].transformer_vocab
        self.indexer = dataset_dict["dataset_reader"]._indexers["tokens"]
        self.max_length = dataset_dict["dataset_reader"].max_length
        self.tokenizer = tokenizer or WordTokenizer()

    def _get_decode_str(
        self,
        token_ids: torch.Tensor
    ):
        decode_str = self.detokenizer.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=True
        )

        return decode_str

    def _get_encode_token_ids(
        self,
        input_str: str
    ):
        tokens = self.detokenizer.tokenize(
            input_str
        )
        token_ids = self.indexer.tokens_to_indices(
            tokens,
            self.transformer_vocab
        )

        return token_ids

    @abc.abstractmethod
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        return NotImplemented

    def _augment(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        # Decode to original string
        decode_str = self._get_decode_str(
            token_ids
        )

        # Tokenize the original string
        decode_tokens = self.tokenizer.tokenize(
            decode_str
        )

        # Action
        augmented_tokens = self._action(
            copy.deepcopy(decode_tokens)
        )

        # Get Augmented String
        augmented_str = self.tokenizer.detokenize(
            augmented_tokens[:self.max_length]
        )

        # Encode to token_ids
        augmented_token_ids = self._get_encode_token_ids(
            augmented_str
        )

        return augmented_token_ids

    def augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
    ):
        augment_text_tensor_list = []
        text_tensor_list = unpad_text_field_tensors(text_field_tensors)

        for text_tensor in text_tensor_list:
            augment_text_tensor = self._augment(text_tensor)
            augment_text_tensor_list.append(augment_text_tensor)

        return move_to_device(
            pad_text_tensor_list(
                augment_text_tensor_list,
                indexer=self.indexer
            ),
            text_tensor_list[0].get_device()
        )


class DeleteAugmenter(Augmenter):
    def __init__(
        self,
        delete_augmenter_params: Dict,
        dataset_dict: Dict
    ):
        super(DeleteAugmenter, self).__init__(
            dataset_dict
        )
        self.magnitude = delete_augmenter_params["magnitude"]

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        if len(tokens) == 1:
            return tokens
        else:
            pass

        # Get Delete Word Indexes
        num_of_del_word = max(1, math.floor(len(tokens) * self.magnitude))
        del_word_idxs = random.sample(range(len(tokens)), num_of_del_word)

        if len(del_word_idxs) > len(tokens) + 1:
            return tokens
        else:
            del_word_idxs.sort()

            # Delete
            for del_word_idx in reversed(del_word_idxs):
                del tokens[del_word_idx]

        return tokens


class SwapAugmenter(Augmenter):
    def __init__(
        self,
        swap_augmenter_params: Dict,
        dataset_dict: Dict
    ):
        super(SwapAugmenter, self).__init__(
            dataset_dict
        )
        self.magnitude = swap_augmenter_params["magnitude"]

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        if len(tokens) == 1:
            return tokens
        else:
            # Sample swap index
            select_idxs = random.sample(range(len(tokens)), max(int(len(tokens) * self.magnitude), 2))
            swap_idxs = copy.deepcopy(select_idxs)

            # Dearangement
            while True:
                safe = True
                for i, j in zip(select_idxs, swap_idxs):
                    if i == j:
                        safe = False
                        random.shuffle(swap_idxs)
                        break
                    else:
                        pass
                if safe:
                    break

            swap_tokens = [copy.deepcopy(tokens[x]) for x in swap_idxs]

            for idx, (select_idx, swap_idx) in enumerate(zip(select_idxs, swap_idxs)):
                tokens[select_idx] = swap_tokens[idx]

            return tokens


class ReplaceAugmenter(Augmenter):
    def __init__(
        self,
        replace_augmenter_params: Dict,
        dataset_dict: Dict
    ):
        super(ReplaceAugmenter, self).__init__(
            dataset_dict
        )
        self.magnitude = replace_augmenter_params["magnitude"]

    def _find_synonyms(
        self,
        token: str
    ):
        synonyms = set()

        for syn in wordnet.synsets(token):
            for synonym_lemma in syn.lemmas():
                synonym = synonym_lemma.name().replace('_', ' ').replace('-', ' ').lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(tuple(synonym.split()))

        synonyms = list(synonyms)

        if tuple([token]) in synonyms:
            synonyms.remove(tuple([token]))
        else:
            pass

        return synonyms

    def _get_synonyms(
        self,
        token: str
    ):
        synonyms = self._find_synonyms(token)

        return synonyms

    def _get_replace_synonym(
        self,
        token: str
    ) -> tuple:
        synonyms = self._get_synonyms(token)

        if synonyms:
            return random.choice(synonyms)
        else:
            return None

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        availiable_token_idxs = []
        availiable_synonyms = []

        for idx in range(len(tokens)):
            replace_synonym = self._get_replace_synonym(
                tokens[idx]
            )

            if replace_synonym:
                availiable_token_idxs.append(idx)
                availiable_synonyms.append(replace_synonym)
            else:
                pass

        if len(availiable_synonyms) == 0:
            return tokens

        replace_idxs = random.sample(
            range(len(availiable_token_idxs)),
            max(int(len(availiable_token_idxs) * self.magnitude), 1)
        )

        for replace_idx in replace_idxs:
            replace_synonym = availiable_synonyms[replace_idx]

            for synonym_token_idx, synonym_token in enumerate(replace_synonym):
                if synonym_token_idx == 0:
                    tokens[availiable_token_idxs[replace_idx]] = synonym_token
                else:
                    tokens.insert(
                        availiable_token_idxs[replace_idx] + synonym_token_idx,
                        synonym_token
                    )

        return tokens


class InsertAugmenter(ReplaceAugmenter):
    def __init__(
        self,
        insert_augmenter_params: Dict,
        dataset_dict: Dict
    ):
        super(InsertAugmenter, self).__init__(
            insert_augmenter_params,
            dataset_dict
        )

    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        availiable_token_idxs = []
        availiable_synonyms = []

        for idx in range(len(tokens)):
            replace_synonym = self._get_replace_synonym(
                tokens[idx]
            )

            if replace_synonym:
                availiable_token_idxs.append(idx)
                availiable_synonyms.append(replace_synonym)
            else:
                pass

        if len(availiable_synonyms) == 0:
            return tokens

        replace_idxs = random.sample(
            range(len(availiable_token_idxs)),
            max(int(len(availiable_token_idxs) * self.magnitude), 1)
        )

        new_tokens = copy.deepcopy(tokens)

        for replace_idx in replace_idxs:
            replace_synonym = availiable_synonyms[replace_idx]
            insert_idx = random.sample(
                range(len(new_tokens)),
                1
            )[0]

            for synonym_token_idx, synonym_token in enumerate(replace_synonym):
                new_tokens.insert(
                    insert_idx + synonym_token_idx,
                    synonym_token
                )

        return new_tokens


class MtTransformers(object):
    def __init__(
        self,
        src_model_name='Helsinki-NLP/opus-mt-en-jap',
        tgt_model_name='Helsinki-NLP/opus-mt-jap-en',
            device='cuda',
            silence=True
    ):
        self.device = 0
        self.src_model_name = src_model_name
        self.tgt_model_name = tgt_model_name
        self.src_model = AutoModelForSeq2SeqLM.from_pretrained(self.src_model_name)
        self.src_model.to(device)
        self.tgt_model = AutoModelForSeq2SeqLM.from_pretrained(self.tgt_model_name)
        self.tgt_model.to(device)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.tgt_model_name)

    def get_device(self):
        return str(self.src_model.device)

    def predict(self, texts, target_words=None, n=1):
        src_tokenized_texts = self.src_tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
        src_translated_ids = self.src_model.generate(**src_tokenized_texts)
        src_translated_texts = self.src_tokenizer.batch_decode(src_translated_ids, skip_special_tokens=True)

        tgt_tokenized_texts = self.tgt_tokenizer(src_translated_texts, padding=True, return_tensors='pt').to(self.device)
        tgt_translated_ids = self.tgt_model.generate(**tgt_tokenized_texts)
        tgt_translated_texts = self.tgt_tokenizer.batch_decode(tgt_translated_ids, skip_special_tokens=True)

        return tgt_translated_texts


class BackTransAugmenter(Augmenter):
    def __init__(
        self,
        backtrans_params: Dict,
        dataset_dict: Dict
    ):
        super(BackTransAugmenter, self).__init__(
            dataset_dict
        )
        self.model = MtTransformers(
            src_model_name=backtrans_params["from_model_name"],
            tgt_model_name=backtrans_params["to_model_name"],
            device=backtrans_params["device"]
        )
        self.magnitude = backtrans_params["magnitude"]

    def _action(
        self,
        text: str
    ) -> str:
        augmented_text = self.model.predict(text)

        return augmented_text[-1]

    def _augment(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        # Decode to original string
        decode_str = self._get_decode_str(
            token_ids
        )

        # Action
        augmented_str = self._action(
            copy.deepcopy(decode_str)
        )

        # Encode to token_ids
        augmented_token_ids = self._get_encode_token_ids(
            augmented_str
        )

        return augmented_token_ids


class IdentityAugmenter(Augmenter):
    def __init__(
        self
    ):
        return

    @overrides
    def _augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
    ):
        return text_field_tensors


def main():
    pass


if __name__ == '__main__':
    main()

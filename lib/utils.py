import torch
import pickle

# from .embedder import UniversalSentenceEmbedder
from tqdm import tqdm
from allennlp.nn.util import move_to_device
from torch.utils.data import DataLoader
from nltk.corpus import wordnet
from typing import Dict, List
from allennlp.data import Vocabulary, DatasetReader, allennlp_collate
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset


def unpad_text_field_tensors(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
    padded_idx: int = 0
) -> List[torch.Tensor]:
    """
    Unpad text field tensors and return a list which lenth is the batch size of text_field_tensor.

    Example:
        Input:
            text_field_tensors = {"tokens": {"tokens": tensor with shape: (B, S)}}
            B = num_of_batch, S = padded sequence
        Output:
            List[tensor with shape (M) * B]
            B = num_of_batch, M = unpadded sequence lenth for different sentence
    """
    text_tensor_list = []
    target_text_field_tensor = text_field_tensors["tokens"]["token_ids"]

    for sent in target_text_field_tensor:
        # sent_len = len(sent) - (sent == padded_idx).sum()
        text_tensor_list.append(sent.clone())

    if len(text_tensor_list) != 1:
        raise ValueError("Augmented but with non-valid batch_size")
    else:
        return text_tensor_list


def pad_text_tensor_list(
    text_tensor_list: List[torch.tensor],
    indexer=None,
    padded_idx: int = 0
):
    """
    Pad list of text tensor back to text_field_tensors with type {Dict[str, Dict[str, torch.Tensor]]}.

    Example:
        Input:
            List[tensor with shape (M) * B]
            B = num_of_batch, M = unpadded sequence lenth for different sentence
        Output:
            text_field_tensors = {"tokens": {"tokens": tensor with shape: (B, S)}}
            B = num_of_batch, S = padded sequence

    """
    padding_length = len(text_tensor_list[0]["token_ids"])
    text_tensor_dict = indexer.as_padded_tensor_dict(
        text_tensor_list[0],  # Allennlp only accept one element [BUG Alert]!
        padding_lengths={
            "token_ids": padding_length,
            "mask": padding_length,
            "type_ids": padding_length
        }
    )

    for key, value in text_tensor_dict.items():
        text_tensor_dict[key] = value.unsqueeze(0)

    return {"tokens": text_tensor_dict}


def add_wordnet_to_vocab(
    vocab: Vocabulary
):
    # iterate over all the possible synomys to enrich vocabulary set
    for syn in wordnet.all_synsets():
        for synonym_lemma in syn.lemmas():
            synonym = synonym_lemma.name().replace('_', ' ').replace('-', ' ').lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])

            for synonym_token in synonym.split():
                vocab.add_token_to_namespace(synonym_token)

    return vocab


def get_sentence_from_text_field_tensors(
    transformer_tokenizer,
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
):
    sentences_token_ids = text_field_tensors["tokens"]["token_ids"].int().tolist()
    return transformer_tokenizer.tokenizer.decode(
        sentences_token_ids[0],
        skip_special_tokens=True
    )


def augment_and_get_texts_from_dataset(
    dataset_reader: DatasetReader,
    dataset: AllennlpDataset,
    reinforcer,
    select_mode: str
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=allennlp_collate)

    augment_texts = []

    for episode_idx, episode in enumerate(dataloader):
        episode = move_to_device(episode, 0)

        # Get augment string from reinforcer
        augment_text = reinforcer.augment(episode)

        augment_texts.append(augment_text)

    return augment_texts


def generate_syntatic_data(
    dataset: AllennlpDataset,
    reinforcer,
    select_mode: str,
    mode_params: Dict,
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=allennlp_collate)

    augment_texts = []

    for episode_idx, episode in tqdm(enumerate(dataloader)):
        episode = move_to_device(episode, 0)

        # Get augment string from reinforcer
        if select_mode == "default":
            reinforcer.policy.load_state_dict(torch.load(mode_params["policy_weight"] + ".pkl"))
            augment_text = reinforcer.augment(episode["text"])
        elif select_mode == "eda":
            augment_text = reinforcer.EDA_augment(episode["text"], mode_params)
        elif select_mode == "backtrans":
            augment_text = reinforcer.BackTrans_augment(episode["text"])

        augment_texts.append(augment_text)

    return augment_texts


def generate_and_save_augmentation_texts(
    policy_weight_paths: List[str],
    saved_names: List[str],
    dataset_reader: DatasetReader,
    train_dataset: AllennlpDataset,
    reinforcer,
    select_mode: str,
):
    for policy_weight_path, saved_name in zip(policy_weight_paths, saved_names):
        import time
        start_time = time.time()

        print("Generating augmented instances with {}".format(policy_weight_path))
        # Load pretrained_weight
        reinforcer.policy.load_state_dict(torch.load(policy_weight_path + ".pkl"))
        # reinforcer.env.USE_embedder = UniversalSentenceEmbedder(
        #     reinforcer.transformer_tokenizer
        # )

        # Get Augmented Sentence
        augmentation_texts = augment_and_get_texts_from_dataset(
            dataset_reader,
            train_dataset,
            reinforcer,
            select_mode
        )

        # Save obj
        save_obj(augmentation_texts, saved_name)

        print("--- %s seconds ---" % (time.time() - start_time))

    return


def save_obj(
    obj: object,
    name: str
):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(
    name: str
):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    pass


if __name__ == '__main__':
    main()

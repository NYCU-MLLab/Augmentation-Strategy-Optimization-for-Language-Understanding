import os
import torch

from typing import Dict
from lib.configurer import get_config_params
from lib.configurer import set_and_get_dataset, set_dataset_vocab
from lib.configurer import set_and_get_text_model, set_and_get_reinforcer, set_and_get_visualizer
from lib.configurer import set_and_get_text_trainer, set_and_get_reinforce_trainer
from lib.configurer import set_and_get_text_dataloader, set_and_get_reinforce_dataloader
from lib.configurer import set_and_save_augmented_texts, set_augments_to_dataset_instances
from lib.configurer import set_and_save_syntatic_data


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_text_model(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module
):
    text_model.train()

    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        mode_params["text_trainer"],
        text_model
    )

    # Get Text DataLoader
    train_dataloader, valid_dataloader, test_dataloader, noisy_dataloader = set_and_get_text_dataloader(
        mode_params["text_trainer"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"],
        noisy_ds=dataset_dict["noisy_ds"]
    )

    # Train Text Model
    text_trainer.fit(
        mode_params["text_trainer"]["epochs"],
        False,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        noisy_dataloader
    )


def train_reinforce_model(
    mode_params: Dict,
    dataset_dict: Dict,
    reinforcer: torch.nn.Module
):
    # Get Reinforce Trainer
    reinforce_trainer = set_and_get_reinforce_trainer(
        mode_params["reinforce_trainer"],
        reinforcer
    )

    # Get Reinforce Dataloader
    reinforce_dataloader = set_and_get_reinforce_dataloader(
        mode_params["reinforce_trainer"]["dataloader"],
        train_ds=dataset_dict["train_ds"]
    )

    # Train Reinforce Dataloader
    reinforce_trainer.fit(
        mode_params["reinforce_trainer"]["epochs"],
        mode_params["reinforce_trainer"]["batch_size"],
        reinforce_dataloader
    )


def generate_augmented_data(
    mode_params: Dict,
    dataset_dict: Dict,
    reinforcer: torch.nn.Module
):
    reinforcer.eval()

    with torch.no_grad():
        set_and_save_augmented_texts(
            mode_params["augmented_instance_generator"],
            dataset_dict["dataset_reader"],
            dataset_dict["train_ds"],
            reinforcer
        )


def finetune_text_model(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module
):
    set_augments_to_dataset_instances(
        dataset_dict,
        mode_params["text_finetuner"]["augmented_instance"]
    )

    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        mode_params["text_finetuner"],
        text_model
    )

    # Get Text DataLoader
    train_dataloader, valid_dataloader, test_dataloader, noisy_dataloader = set_and_get_text_dataloader(
        mode_params["text_finetuner"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"],
        noisy_ds=dataset_dict["noisy_ds"]
    )

    # Train Text Model
    text_model.train()
    text_model.set_augment_field_names(
        mode_params["text_finetuner"]["augmented_instance"]
    )
    text_trainer.fit(
        mode_params["text_finetuner"]["epochs"],
        True,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        noisy_dataloader
    )


def load_pretrained_text_model(
    mode_params: Dict,
    text_model: torch.nn.Module
):
    print("Loading pretrained weight for embedder ...")
    text_model.embedder.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["embedder"]
        )
    )

    print("Loading pretrained weight for encoder ...")
    text_model.encoder.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["encoder"]
        )
    )

    print("Loading pretrained weight for classifier ...")
    text_model.classifier.load_state_dict(
        torch.load(
            mode_params["pretrained_text_model"]["classifier"]
        )
    )


def train_with_augmented_data_from_scratch(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module
):
    set_augments_to_dataset_instances(
        dataset_dict,
        mode_params["text_trainer"]["augmented_instance"]
    )

    # Get Text Trainer
    text_trainer = set_and_get_text_trainer(
        mode_params["text_trainer"],
        text_model
    )

    # Get Text DataLoader
    train_dataloader, valid_dataloader, test_dataloader = set_and_get_text_dataloader(
        mode_params["text_trainer"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"]
    )

    # Train Text Model
    text_model.train()
    text_model.set_augment_field_names(
        mode_params["text_trainer"]["augmented_instance"]
    )
    text_trainer.fit(
        mode_params["text_trainer"]["epochs"],
        True,
        train_dataloader,
        valid_dataloader,
        test_dataloader
    )


def all_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    train_text_model(
        mode_params,
        dataset_dict,
        text_model
    )

    train_reinforce_model(
        mode_params,
        dataset_dict,
        reinforcer
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def all_procedure_with_pretrained_text(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    train_reinforce_model(
        mode_params,
        dataset_dict,
        reinforcer
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def all_procedure_with_all_pretrained(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    generate_augmented_data(
        mode_params,
        dataset_dict,
        reinforcer
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def finetune_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    reinforcer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    finetune_text_model(
        mode_params,
        dataset_dict,
        text_model
    )


def visualize_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model: torch.nn.Module,
    visualizer: torch.nn.Module
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    # Set pretrained embedding to visualizer
    visualizer.enmbedder = text_model.embedder
    visualizer.encoder = text_model.encoder
    del text_model

    # Standard Visualize
    visualizer.visualize(
        mode_params["visualizer"],
        dataset_dict["train_ds"]
    )


def syntatic_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    reinforcer
):
    set_and_save_syntatic_data(
        mode_params["syntatic"],
        dataset_dict,
        reinforcer
    )


def test_procedure(
    mode_params: Dict,
    dataset_dict: Dict,
    text_model
):
    load_pretrained_text_model(
        mode_params,
        text_model
    )

    train_dataloader, valid_dataloader, test_dataloader = set_and_get_text_dataloader(
        mode_params["text_finetuner"]["dataloader"],
        train_ds=dataset_dict["train_ds"],
        valid_ds=dataset_dict["valid_ds"],
        test_ds=dataset_dict["test_ds"]
    )

    text_trainer = set_and_get_text_trainer(
        mode_params["text_finetuner"],
        text_model
    )

    text_trainer.predict(
        test_dataloader
    )


def main(config_params):
    # Get Dataset
    dataset_dict = set_and_get_dataset(
        config_params["dataset"]
    )

    # Set Dataset Vocab
    set_dataset_vocab(
        dataset_dict
    )

    # Get Text Model
    text_model = set_and_get_text_model(
        config_params["text_model"],
        dataset_dict
    )

    # Get Reinforcer
    reinforcer = set_and_get_reinforcer(
        config_params["reinforcer"],
        dataset_dict,
        text_model
    )

    # Move to GPU
    if config_params["env"]["USE_GPU"] is not None:
        text_model = text_model.cuda(config_params["env"]["USE_GPU"])
        reinforcer = reinforcer.cuda(config_params["env"]["USE_GPU"])
    else:
        pass

    # Go to Mode
    if config_params["train_mode"]["select_mode"] == 0:
        all_procedure(
            config_params["train_mode"]["0"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 1:
        all_procedure_with_pretrained_text(
            config_params["train_mode"]["1"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 2:
        all_procedure_with_all_pretrained(
            config_params["train_mode"]["2"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 3:
        finetune_procedure(
            config_params["train_mode"]["3"],
            dataset_dict,
            text_model,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 4:
        train_with_augmented_data_from_scratch(
            config_params["train_mode"]["4"],
            dataset_dict,
            text_model
        )
    elif config_params["train_mode"]["select_mode"] == 5:
        visualizer = set_and_get_visualizer(
            config_params["visualizer"],
            text_model,
            dataset_dict["dataset_vocab"]
        )
        visualize_procedure(
            config_params["train_mode"]["4"],
            dataset_dict,
            text_model,
            visualizer
        )
    elif config_params["train_mode"]["select_mode"] == 6:
        syntatic_procedure(
            config_params["train_mode"]["6"],
            dataset_dict,
            reinforcer
        )
    elif config_params["train_mode"]["select_mode"] == 7:
        test_procedure(
            config_params["train_mode"]["7"],
            dataset_dict,
            text_model
        )
    else:
        raise ValueError


if __name__ == '__main__':
    main(get_config_params("model_config.json"))

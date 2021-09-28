import textattack
import transformers
import pandas as pd
from argparse import ArgumentParser


NUM_CLASSES = 2


class AugmentArgs:
    input_csv: str
    output_csv: str
    input_column: str
    overwrite: bool = True

    @classmethod
    def _add_parser_args(cls, parser):
        parser.add_argument(
            "--datapath-prefix",
            required=True,
            type=str,
            help="Path of input dataset to attack.",
        )

        parser.add_argument(
            "--attack-method",
            required=True,
            type=str,
            help="Name of attack recipe."
        )

        parser.add_argument(
            "--target-model",
            required=False,
            type=str,
            default="roberta-base"
        )


def get_dataset(
    datapath: str
):
    input_df = pd.read_csv(datapath)
    sentences = input_df["sentence"].to_list()
    sentences = [str(sentence) for sentence in sentences]
    labels = input_df["label"].to_list()
    pre_dataset = list(zip(sentences, labels))

    global NUM_CLASSES
    NUM_CLASSES = len(set(labels))

    return textattack.datasets.Dataset(pre_dataset)


def get_attack_module(
    attack_method_name: str,
    model_wrapper
):
    if attack_method_name == "pwws":
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    elif attack_method_name == "fast-alzantot":
        attack = textattack.attack_recipes.FasterGeneticAlgorithmJia2019.build(model_wrapper)
    elif attack_method_name == "iga":
        attack = textattack.attack_recipes.IGAWang2019.build(model_wrapper)
    elif attack_method_name == "textfooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif attack_method_name == "hotflip":
        attack = textattack.attack_recipes.HotFlipEbrahimi2017.build(model_wrapper)
    elif attack_method_name == "bae":
        attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
    elif attack_method_name == "deepwordbug":
        attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
    elif attack_method_name == "input-reduction":
        attack = textattack.attack_recipes.InputReductionFeng2018.build(model_wrapper)
    elif attack_method_name == "kuleshov":
        attack = textattack.attack_recipes.Kuleshov2017.build(model_wrapper)
    elif attack_method_name == "pso":
        attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)
    elif attack_method_name == "textbugger":
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    else:
        raise ValueError("Unsupported Attack Method")

    return attack


def main():
    parser = ArgumentParser()
    AugmentArgs._add_parser_args(parser)
    args = parser.parse_args()

    train_ds = get_dataset(args.datapath_prefix + "_train.csv")
    test_ds = get_dataset(args.datapath_prefix + "_test.csv")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.target_model, num_labels=NUM_CLASSES)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    attack = get_attack_module(args.attack_method, model_wrapper)

    training_args = textattack.TrainingArgs(
        num_epochs=9,
        num_clean_epochs=3,
        attack_epoch_interval=2,
        num_train_adv_examples=6000,
        learning_rate=1e-5,
        num_warmup_steps=0.06,
        attack_num_workers_per_device=9,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_to_tb=True,
    )
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_ds,
        test_ds,
        training_args
    )

    trainer.train()

    noisy_ds = []
    noisy_names = ["stack_eda", "eda", "embedding", "clare", "checklist", "char", "backtrans_de", "backtrans_ru", "backtrans_zh", "spell"]
    noisy_ds.append(get_dataset(args.datapath_prefix + "_stack_eda.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_eda.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_embedding.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_clare.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_checklist.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_char.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_backtrans_de.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_backtrans_ru.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_backtrans_zh.csv"))
    noisy_ds.append(get_dataset(args.datapath_prefix + "_spell.csv"))

    for idx, nds in enumerate(noisy_ds):
        trainer = textattack.Trainer(
            model_wrapper,
            "classification",
            attack,
            train_ds,
            nds,
            training_args
        )
        print(noisy_names[idx])
        trainer.evaluate()


if __name__ == '__main__':
    main()

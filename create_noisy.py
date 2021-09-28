import pandas as pd
import nlpaug.augmenter.word as naw

from tqdm import tqdm
from argparse import ArgumentParser


class AugmentArgs:
    input_csv: str
    output_csv: str
    input_column: str
    overwrite: bool = True

    @classmethod
    def _add_parser_args(cls, parser):
        parser.add_argument(
            "--input-csv",
            required=True,
            type=str,
            help="Path of input CSV file to augment.",
        )
        parser.add_argument(
            "--output-csv",
            required=True,
            type=str,
            help="Path of CSV file to output augmented data.",
        )
        parser.add_argument(
            "--input-column",
            "--i",
            required=True,
            type=str,
            help="CSV input column to be augmented",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action="store_true",
            help="overwrite output file, if it exists",
        )
        parser.add_argument(
            "--language",
            default="de",
            type=str,
            help="Backtranslation will use"
        )
        parser.add_argument(
            "--recipe",
            required=True,
            type=str,
            help="Augmentation method"
        )
        parser.add_argument(
            "--transformations-per-example",
            type=int,
            help="Number of transformations per example"
        )


def initialize_augmenter(
    input_args
):
    recipe_name = input_args.recipe

    if recipe_name == "spell":
        augmenter = naw.SpellingAug()
    elif recipe_name == "backtrans":
        if input_args.language == "de":
            augmenter = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de',
                to_model_name='facebook/wmt19-de-en',
                device="cuda",
                verbose=10,
                batch_size=128
            )
        elif input_args.language == "ru":
            augmenter = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-ru',
                to_model_name='facebook/wmt19-ru-en',
                device="cuda",
                verbose=10,
                batch_size=128
            )
        elif input_args.language == "zh":
            augmenter = naw.BackTranslationAug(
                from_model_name="Helsinki-NLP/opus-mt-en-zh",
                to_model_name="Helsinki-NLP/opus-mt-zh-en",
                device="cuda",
                verbose=10,
                batch_size=128
            )
        else:
            raise KeyError("Unsupport Language")
    else:
        raise KeyError("Unsupport Augmentation")

    return augmenter


class Augmenter:
    def __init__(
        self,
        input_args
    ):
        self.augmenter = initialize_augmenter(input_args)
        self.input_csv = pd.read_csv(input_args.input_csv)
        self.output_csv_datapath = input_args.output_csv
        self.target_column = input_args.input_column
        self.transformation_nums = input_args.transformations_per_example

    def augment(
        self
    ):
        columns = self.input_csv.columns.to_list()
        augment_dict = {key: [] for key in columns}

        print(self.input_csv)

        for index, row in tqdm(self.input_csv.iterrows()):
            augment_texts = self.augmenter.augment(row[self.target_column], n=self.transformation_nums)
            if type(augment_texts) == str:
                augment_texts = [augment_texts]
            else:
                pass
            augment_dict[self.target_column] += augment_texts
            for name in [n for n in columns if n != self.target_column]:
                augment_dict[name] += [row[name]] * len(augment_texts)

        output_df = pd.DataFrame(augment_dict)
        output_df.to_csv(self.output_csv_datapath, index=False)

        print(output_df)


def main():
    # create parser
    parser = ArgumentParser()
    AugmentArgs._add_parser_args(parser)
    args = parser.parse_args()

    augmenter = Augmenter(args)
    augmenter.augment()


if __name__ == '__main__':
    main()

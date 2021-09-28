import logging
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from .tokenizer import WordTokenizer
from .utils import load_obj

from typing import Dict, Optional
from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import LabelField, TextField, Field

logger = logging.getLogger(__name__)


@DatasetReader.register("CR")
class SentimentDatasetReader(DatasetReader):
    def __init__(
        self,
        dataset_params: Dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._indexers = {
            "tokens": PretrainedTransformerIndexer(
                dataset_params["transformer_model_name"]
            )
        }
        self.transformer_tokenizer = PretrainedTransformerTokenizer(
            dataset_params["transformer_model_name"]
        )
        self.transformer_vocab = Vocabulary.from_pretrained_transformer(
            dataset_params["transformer_model_name"]
        )
        self.detokenizer = WordTokenizer()
        self.max_length = dataset_params["max_length"]

    @overrides
    def _read(self, file_path):
        corpus = pd.read_csv(
            file_path
        )
        reviews, labels = list(corpus.sentence), list(corpus.label)

        for review, label in zip(reviews, labels):
            if type(review) != str:
                if math.isnan(review):
                    review = "."

            instance = self.text_to_instance(review, str(label))

            if instance is not None:
                yield instance
            else:
                pass

    @overrides
    def text_to_instance(
        self,
        text: str,
        sentiment: str = None
    ) -> Optional[Instance]:
        tokens = self.transformer_tokenizer.tokenize(text)

        text_field = TextField(
            tokens,
            token_indexers=self._indexers
        )
        fields: Dict[str, Field] = {
            "text": text_field
        }

        if sentiment is not None:
            fields["label"] = LabelField(sentiment)
        else:
            pass

        return Instance(fields)

    def get_token_indexers(self):
        return self._token_indexers


@DatasetReader.register("sst_tokens")
class StanfordSentimentTreeBankDatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.
    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).
    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    Registered as a `DatasetReader` with name "sst_tokens".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    use_subtrees : `bool`, optional, (default = `False`)
        Whether or not to use sentiment-tagged subtrees.
    granularity : `str`, optional (default = `"5-class"`)
        One of `"5-class"`, `"3-class"`, or `"2-class"`, indicating the number
        of sentiment labels to use.
    """

    def __init__(
        self,
        sst_params: Dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._indexers = {
            "tokens": PretrainedTransformerIndexer(
                sst_params["transformer_model_name"]
            )
        }

        self.transformer_tokenizer = PretrainedTransformerTokenizer(
            sst_params["transformer_model_name"]
        )

        self.transformer_vocab = Vocabulary.from_pretrained_transformer(
            sst_params["transformer_model_name"]
        )

        self._use_subtrees = sst_params["use_subtrees"]
        self.detokenizer = WordTokenizer()
        self.max_length = sst_params["max_length"]
        self.robust_test = False
        if sst_params["noise_datapath"] != "none":
            self.noise_data = load_obj(sst_params["noise_datapath"])
        else:
            self.noise_data = None

        allowed_granularities = ["5-class", "3-class", "2-class"]

        if sst_params["granularity"] not in allowed_granularities:
            raise ConfigurationError(
                "granularity is {}, but expected one of: {}".format(
                    sst_params["granularity"], allowed_granularities
                )
            )
        self._granularity = sst_params["granularity"]

    @overrides
    def _read(self, file_path):
        sentences = []
        labels = []

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for idx, line in enumerate(data_file.readlines()):
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                if self._use_subtrees:
                    for subtree in parsed_line.subtrees():
                        instance = self.text_to_instance(
                            self.detokenizer.detokenize(subtree.leaves()),
                            subtree.label()
                        )
                        if instance is not None:
                            yield instance
                else:
                    if self.robust_test is True and self.noise_data is not None:
                        text = self.noise_data[idx]
                    else:
                        text = self.detokenizer.detokenize(parsed_line.leaves())

                    instance = self.text_to_instance(
                        text,
                        parsed_line.label()
                    )

                    if self._granularity == "2-class":
                        label = int(parsed_line.label())
                        if label > 2:
                            label = "1"
                            sentences.append(text)
                            labels.append(label)
                        elif label < 2:
                            label = "0"
                            sentences.append(text)
                            labels.append(label)
                        else:
                            pass
                    else:
                        sentences.append(text)
                        labels.append(parsed_line.label())

                    if instance is not None:
                        yield instance

        input_df = pd.DataFrame({"sentence": sentences, "label": labels}, dtype=object)
        input_df.to_csv(file_path+".csv", index=False)

    @overrides
    def text_to_instance(
        self,
        text: str,
        sentiment: str = None
    ) -> Optional[Instance]:
        tokens = self.transformer_tokenizer.tokenize(text)

        text_field = TextField(
            tokens[:self.max_length],
            token_indexers=self._indexers
        )
        fields: Dict[str, Field] = {
            "text": text_field
        }

        if sentiment is not None:
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
            fields["label"] = LabelField(sentiment)
        else:
            pass

        return Instance(fields)

    def get_token_indexers(self):
        return self._token_indexers


def get_sst_ds(
    sst_params: Dict,
    train_data_path="data/sst/train.txt",
    valid_data_path="data/sst/dev.txt",
    test_data_path="data/sst/test.txt",
):
    sst_dataset_reader = StanfordSentimentTreeBankDatasetReader(
        sst_params
    )

    train_ds = sst_dataset_reader.read(train_data_path)

    if sst_params["proportion"] != 1:
        import random
        random.seed(2003)
        train_ds.instances = random.sample(
            train_ds.instances,
            int(len(train_ds.instances) * sst_params["proportion"])
        )
    else:
        pass
    valid_ds = sst_dataset_reader.read(valid_data_path)
    sst_dataset_reader.robust_test = True
    test_ds = sst_dataset_reader.read(test_data_path)

    return train_ds, valid_ds, test_ds, sst_dataset_reader


def split_dataset(
    all_data_path: str
):
    corpus = pd.read_pickle(all_data_path)
    train, test = train_test_split(
        corpus,
        test_size=0.25,
        random_state=2003
    )
    train, valid = train_test_split(
        train,
        test_size=0.1,
        random_state=2003
    )

    train.to_pickle(all_data_path + "train")
    valid.to_pickle(all_data_path + "valid")
    test.to_pickle(all_data_path + "test")


def get_sentimenmt_ds(
    dataset_params: Dict
):
    # split_dataset(
    #     dataset_params["datapath"]
    # )

    sentiment_dataset_reader = SentimentDatasetReader(dataset_params)

    train_ds = sentiment_dataset_reader.read(
        dataset_params["datapath"] + "_train.csv"
    )
    valid_ds = sentiment_dataset_reader.read(
        dataset_params["datapath"] + "_valid.csv"
    )
    test_ds = sentiment_dataset_reader.read(
        dataset_params["datapath"] + "_test.csv"
    )

    if dataset_params["noisy_datapath"] == "none":
        noisy_ds = None
    elif dataset_params["noisy_datapath"] == "all":
        stack_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_stack_eda.csv"
        )
        eda_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_eda.csv"
        )
        embedding_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_embedding.csv"
        )
        clare_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_clare.csv"
        )
        check_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_checklist.csv"
        )
        char_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_char.csv"
        )
        de_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_backtrans_de.csv"
        )
        ru_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_backtrans_ru.csv"
        )
        zh_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_backtrans_zh.csv"
        )
        spell_ds = sentiment_dataset_reader.read(
            dataset_params["datapath"] + "_spell.csv"
        )
        noisy_ds = [stack_ds, eda_ds, embedding_ds, clare_ds, check_ds, char_ds, de_ds, ru_ds, zh_ds, spell_ds]

    else:
        noisy_ds = [sentiment_dataset_reader.read(
            dataset_params["noisy_datapath"]
        )]

    return train_ds, valid_ds, test_ds, noisy_ds, sentiment_dataset_reader


def main():
    pass


if __name__ == '__main__':
    main()

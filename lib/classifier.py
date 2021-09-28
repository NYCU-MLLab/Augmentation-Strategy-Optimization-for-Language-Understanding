import torch

from typing import Dict
from overrides import overrides
from allennlp.modules import FeedForward  # noqa


class TextClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        feedforward_params: Dict
    ):
        super(TextClassifier, self).__init__()

        # feedforward = FeedForward(
        #     input_dim=input_size,
        #     num_layers=feedforward_params["num_layers"],
        #     hidden_dims=feedforward_params["hidden_dims"],
        #     activations=feedforward_params["activations"],
        #     dropout=feedforward_params["dropout"]
        # )

        final_linear = torch.nn.Linear(
            # feedforward.get_output_dim(),
            input_size,
            output_size
        )

        self.classifiers = torch.nn.ModuleList(
            [final_linear]
        )

        self.output_size = output_size

    @overrides
    def forward(
        self,
        encode_X
    ):
        for classifier in self.classifiers:
            encode_X = classifier(encode_X)

        return encode_X

    def get_output_dim(
        self
    ) -> int:
        return self.output_size


def main():
    pass


if __name__ == '__main__':
    main()

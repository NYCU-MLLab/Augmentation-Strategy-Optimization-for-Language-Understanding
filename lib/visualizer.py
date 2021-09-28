import abc
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .encoder import TextEncoder
from .embedder import TextEmbedder
from .utils import load_obj
from tqdm import tqdm
from typing import Dict
from overrides import overrides
from sklearn.manifold import TSNE, Isomap
from torch.utils.data import DataLoader
from allennlp.data import allennlp_collate
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import move_to_device


def get_augmented_instances(
    augmented_instances_save_name: str
):
    augmented_instances = load_obj(augmented_instances_save_name)

    return augmented_instances


class visualizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _reduce_dim(
        input_X: torch.Tensor
    ):
        return NotImplemented

    @abc.abstractmethod
    def _encode(
        token_X: torch.Tensor
    ):
        return NotImplemented

    @abc.abstractmethod
    def visualize(
        self,
        ds: AllennlpDataset
    ):
        return NotImplemented


class IsomapVisualizer(visualizer):
    def __init__(
        self,
        visualizer_params: Dict,
        embedder: TextEmbedder,
        encoder: TextEncoder,
        vocab: Vocabulary
    ):
        self.embedder = embedder
        self.encoder = encoder
        self.vocab = vocab
        self.col_names = visualizer_params["col_names"]
        self.reducer = Isomap(
            n_neighbors=visualizer_params["n_neighbors"],
            n_components=visualizer_params["n_components"],
            n_jobs=visualizer_params["n_jobs"]
        )
        self.save_fig_path = visualizer_params["save_fig_path"] + "fig"
        self.GPU = next(encoder.parameters()).get_device()

    def _get_sample_array_dict(
        self,
        encoded_array_X_dict: Dict,
        encoded_array_Y_dict: Dict,
        proportion: float
    ):
        sample_idx = np.random.choice(
            encoded_array_X_dict["origin"].shape[0],
            int(encoded_array_X_dict["origin"].shape[0] * proportion),
            replace=False
        )

        for key, value in encoded_array_X_dict.items():
            encoded_array_X_dict[key] = value[sample_idx, :]
        for key, value in encoded_array_Y_dict.items():
            encoded_array_Y_dict[key] = value[sample_idx]

        return encoded_array_X_dict, encoded_array_Y_dict

    def _encode(
        self,
        token_X: torch.Tensor
    ):
        with torch.no_grad():
            embed_X = self.embedder(token_X)
            tokens_mask = get_text_field_mask(token_X)
            encode_X = self.encoder(embed_X.detach(), tokens_mask)

        return encode_X

    def _get_encoded_array(
        self,
        dataloader: DataLoader
    ):
        total_X = []
        total_Y = []

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if self.GPU > -1:
                batch = move_to_device(batch, self.GPU)
            else:
                pass

            token_X = batch["tokens"]
            label_Y = batch["label"].detach().cpu().numpy()

            encode_X = self._encode(token_X).cpu().numpy()

            total_X.append(encode_X)
            total_Y.append(label_Y)

        total_X = np.vstack(total_X)
        total_Y = np.concatenate(total_Y)

        return total_X, total_Y

    def _get_encoded_array_dict(
        self,
        visualizer_params: Dict,
        ds: AllennlpDataset
    ):
        encoded_array_X_dict = {}
        encoded_array_Y_dict = {}

        # Get original encoded array first
        dataloader = DataLoader(
            ds,
            batch_size=visualizer_params["batch_size"],
            shuffle=False,
            collate_fn=allennlp_collate
        )

        total_X, total_Y = self._get_encoded_array(
            dataloader
        )

        encoded_array_X_dict["origin"] = total_X
        encoded_array_Y_dict["origin"] = total_Y

        # Get encoded array for different augmented policy
        for augmented_path in visualizer_params["augmented_instance"]:
            augmented_instances = get_augmented_instances(
                augmented_path
            )
            ds.instances = augmented_instances

            dataloader = DataLoader(
                ds,
                batch_size=visualizer_params["batch_size"],
                shuffle=False,
                collate_fn=allennlp_collate
            )

            total_X, total_Y = self._get_encoded_array(
                dataloader
            )
            encoded_array_X_dict[augmented_path] = total_X
            encoded_array_Y_dict[augmented_path] = total_Y

        return {"X": encoded_array_X_dict, "Y": encoded_array_Y_dict}

    @overrides
    def _reduce_dim(
        self,
        input_X: torch.Tensor
    ):
        reduce_X = self.reducer.transform(input_X)

        return reduce_X

    def _get_plot_df(
        self,
        origin_X: np.ndarray,
        augment_X: np.ndarray,
        origin_Y: np.ndarray,
        augment_Y: np.ndarray,
    ):
        if np.array_equal(origin_X, augment_X):
            is_origin = True
        else:
            is_origin = False

        total_X = np.vstack([origin_X, augment_X])

        origin_Y = np.array(
            list(
                map(
                    lambda x: self.vocab.get_token_from_index(
                        x,
                        namespace="labels"
                    ),
                    list(origin_Y)
                )
            )
        )

        if is_origin:
            augment_Y = origin_Y
        else:
            augment_Y = np.array(
                list(
                    map(
                        lambda x: self.vocab.get_token_from_index(
                            x,
                            namespace="labels"
                        ) + "_aug",
                        list(augment_Y)
                    )
                )
            )

        # Union
        union_Y = np.concatenate([origin_Y, augment_Y])
        union_Y = union_Y.reshape([union_Y.shape[0], 1])
        del origin_Y, augment_Y
        union_arr = np.hstack([total_X, union_Y])
        del total_X, union_Y

        # Get Df
        plot_df = pd.DataFrame(
            union_arr,
            columns=self.col_names
        ).astype({
            self.col_names[0]: "float32",
            self.col_names[1]: "float32",
            self.col_names[2]: "category"
        })
        del union_arr

        return plot_df

    def _plot(
        self,
        plot_df: pd.DataFrame,
        meta_filename: str,
        is_comparison: bool = False
    ):
        palette = sns.color_palette("Paired")

        if plot_df[self.col_names[2]].nunique() == 1:
            palette = [
                palette[1]
            ]
            sizes = [
                10
            ]
            markers = [
                "o"
            ]
        elif plot_df[self.col_names[2]].nunique() == 2:
            palette = [
                palette[1],
                palette[5],
            ]
            sizes = [
                10,
                10
            ]
            markers = [
                "o",
                "s"
            ]
        elif plot_df[self.col_names[2]].nunique() == 4:
            palette = [
                palette[1],
                palette[0],
                palette[5],
                palette[4]
            ]
            sizes = [
                10,
                5,
                10,
                5
            ]
            markers = [
                "o",
                "P",
                "s",
                "X"
            ]
        elif plot_df[self.col_names[2]].nunique() == 5:
            palette = [
                palette[1],
                palette[3],
                palette[5],
                palette[7],
                palette[9]
            ]
            sizes = [
                10,
                10,
                10,
                10,
                10
            ]
            markers = [
                "o",
                "o",
                "o",
                "o",
                "o"
            ]
        elif plot_df[self.col_names[2]].nunique() == 10:
            palette = [
                palette[1],
                palette[0],
                palette[3],
                palette[2],
                palette[5],
                palette[4],
                palette[7],
                palette[6],
                palette[9],
                palette[8]
            ]
            sizes = [
                10,
                5,
                10,
                5,
                10,
                5,
                10,
                5,
                10,
                5
            ]
            markers = [
                "o",
                "P",
                "o",
                "P",
                "o",
                "P",
                "o",
                "P",
                "o",
                "P"
            ]

        if is_comparison is True:
            label_num = plot_df[self.col_names[2]].nunique()
            palette = sns.color_palette("hls", label_num)
            sizes = [10] * label_num
            sizes[0] = 90
            markers = ["o"] * label_num
            markers[0] = "X"
        else:
            pass

        # plt.xlim(-80, 80)
        # plt.ylim(-80, 80)

        sns.scatterplot(
            data=plot_df,
            x=self.col_names[0],
            y=self.col_names[1],
            size=self.col_names[2],
            size_order=list(plot_df[self.col_names[2]].unique()).sort(),
            sizes=sizes,
            hue=self.col_names[2],
            hue_order=list(plot_df[self.col_names[2]].unique()).sort(),
            style=self.col_names[2],
            style_order=list(plot_df[self.col_names[2]].unique()).sort(),
            markers=markers,
            legend="full",
            palette=palette,
            alpha=1,
            s=30
        )

        import os

        plt.legend(loc='upper right')

        plt.savefig(
            self.save_fig_path + '_' + meta_filename.split(os.sep)[-1],
            dpi=1200
        )
        plt.close()

    def _plot_single(
        self,
        plot_df: pd.DataFrame,
        meta_filename: str
    ):
        palette = sns.color_palette("Paired")

        label_name = plot_df[self.col_names[2]][0]
        label_num = plot_df[self.col_names[2]].nunique()

        if label_name == "0" and label_num == 1:
            palette = [
                palette[1]
            ]
            sizes = [
                10
            ]
            markers = [
                "o"
            ]
        elif label_name == "1" and label_num == 1:
            palette = [
                palette[5]
            ]
            sizes = [
                10
            ]
            markers = [
                "s"
            ]
        elif label_name == "0" and label_num == 2:
            palette = [
                palette[1],
                palette[0]
            ]
            sizes = [
                10,
                5
            ]
            markers = [
                "o",
                "P"
            ]
        elif label_name == "1" and label_num == 2:
            palette = [
                palette[5],
                palette[4]
            ]
            sizes = [
                10,
                5
            ]
            markers = [
                "s",
                "X"
            ]
        else:
            palette = sns.color_palette("Paired", label_num)
            sizes = [10] * label_num
            markers = ["s"] * label_num

        # plt.xlim(-80, 80)
        # plt.ylim(-80, 80)

        sns.scatterplot(
            data=plot_df,
            x=self.col_names[0],
            y=self.col_names[1],
            size=self.col_names[2],
            size_order=list(plot_df[self.col_names[2]].unique()).sort(),
            sizes=sizes,
            hue=self.col_names[2],
            hue_order=list(plot_df[self.col_names[2]].unique()).sort(),
            style=self.col_names[2],
            style_order=list(plot_df[self.col_names[2]].unique()).sort(),
            markers=markers,
            legend="full",
            palette=palette,
            alpha=1,
            s=30
        )

        import os

        plt.legend(loc='upper right')

        plt.savefig(
            self.save_fig_path + '_' + meta_filename.split(os.sep)[-1],
            dpi=1200
        )
        plt.close()

    def standard_visualize(
        self,
        visualizer_params: Dict,
        ds: AllennlpDataset
    ):
        encoded_array_dict = self._get_encoded_array_dict(
            visualizer_params,
            ds
        )

        self.reducer.fit(encoded_array_dict["X"]["origin"])

        encoded_array_X_dict, encoded_array_Y_dict = self._get_sample_array_dict(
            encoded_array_dict["X"],
            encoded_array_dict["Y"],
            visualizer_params["proportion"]
        )
        encoded_array_dict["X"] = encoded_array_X_dict
        encoded_array_dict["Y"] = encoded_array_Y_dict

        for key, value in encoded_array_dict["X"].items():
            encoded_array_dict["X"][key] = self.reducer.transform(value)

        # Single Point Sample
        sample_idx = np.random.choice(
            encoded_array_dict["X"]["origin"].shape[0],
            1,
            replace=False
        )

        total_X = []
        total_Y = []

        for key, value in encoded_array_dict["X"].items():
            origin_Y = encoded_array_dict["Y"]["origin"]
            augment_Y = encoded_array_dict["Y"][key]
            s_origin_Y = encoded_array_dict["Y"]["origin"][sample_idx]
            s_augment_Y = encoded_array_dict["Y"][key][sample_idx]

            # Plot all points
            plot_df = self._get_plot_df(
                encoded_array_dict["X"]["origin"],
                value,
                origin_Y,
                augment_Y
            )
            self._plot(
                plot_df,
                key
            )

            # Plot single point
            plot_df = self._get_plot_df(
                encoded_array_dict["X"]["origin"][sample_idx, :],
                value[sample_idx, :],
                s_origin_Y,
                s_augment_Y
            )
            # self._plot_single(
            #     plot_df,
            #     key + "_single"
            # )

            # Collect All point
            total_X.append(value[sample_idx, :])
            total_Y.append(s_augment_Y)

        # Plot all points
        plot_df = self._get_plot_df(
            encoded_array_dict["X"]["origin"][sample_idx, :],
            np.vstack(total_X[1:]),
            encoded_array_dict["Y"]["origin"][sample_idx],
            np.concatenate(total_Y[1:])
        )
        self._plot_single(
            plot_df,
            "all"
        )

    def _get_comparison_encoded_array_dict(
        self,
        visualizer_params: Dict,
        ds: AllennlpDataset
    ):
        comparison_dict = {}
        origin_dict = {}

        # Get original encoded array first
        dataloader = DataLoader(
            ds,
            batch_size=visualizer_params["batch_size"],
            shuffle=False,
            collate_fn=allennlp_collate
        )

        origin_origin_dict = {}
        total_X, total_Y = self._get_encoded_array(
            dataloader
        )
        origin_origin_dict["X"] = total_X
        origin_origin_dict["Y"] = total_Y
        origin_dict["origin"] = origin_origin_dict

        comparison_dict["origin"] = origin_dict

        for comp_key, comp_val in visualizer_params["sim_comparison"]["comparison_dict"].items():
            comp_dict = {}
            for augmented_path in comp_val:
                aug_dict = {}
                augmented_instances = get_augmented_instances(
                    augmented_path
                )
                ds.instances = augmented_instances

                dataloader = DataLoader(
                    ds,
                    batch_size=visualizer_params["batch_size"],
                    shuffle=False,
                    collate_fn=allennlp_collate
                )

                total_X, total_Y = self._get_encoded_array(
                    dataloader
                )

                aug_dict["X"] = total_X
                aug_dict["Y"] = total_Y
                comp_dict[augmented_path] = aug_dict
            comparison_dict[comp_key] = comp_dict

        return comparison_dict

    def _get_comp_plot_df(
        self,
        comparison_dict: Dict
    ):
        total_X = []
        total_Y = []

        for comp_key, comp_val in comparison_dict.items():
            total_X.append(comp_val["X"])
            Y = np.array(
                list(
                    map(
                        lambda x: self.vocab.get_token_from_index(
                            x,
                            namespace="labels"
                        ) + "_" + comp_key,
                        list(comp_val["Y"])
                    )
                )
            )
            total_Y.append(Y)

        # Union
        union_X = np.vstack(total_X)
        union_Y = np.concatenate(total_Y)
        union_Y = union_Y.reshape([union_Y.shape[0], 1])
        del total_X, total_Y
        union_arr = np.hstack([union_X, union_Y])
        del union_X, union_Y

        # Get Df
        plot_df = pd.DataFrame(
            union_arr,
            columns=self.col_names
        ).astype({
            self.col_names[0]: "float32",
            self.col_names[1]: "float32",
            self.col_names[2]: "category"
        })
        del union_arr

        return plot_df

    def comparison_visualize(
        self,
        visualizer_params: Dict,
        ds
    ):
        encoded_array_dict = self._get_comparison_encoded_array_dict(
            visualizer_params,
            ds
        )

        sample_idx = np.random.choice(
            encoded_array_dict["origin"]["origin"]["X"].shape[0],
            1
        )

        comparison_dict = {}

        for comp_key, comp_val in encoded_array_dict.items():
            comp_X = []
            comp_Y = []
            comp_dict = {}
            for select_key, select_val in comp_val.items():
                comp_X.append(
                    self.reducer.transform(
                        select_val["X"][sample_idx, :]
                    )
                )
                comp_Y.append(select_val["Y"][sample_idx])
            comp_dict["X"] = np.vstack(comp_X)
            comp_dict["Y"] = np.concatenate(comp_Y)
            comparison_dict[comp_key] = comp_dict

        del encoded_array_dict

        plot_df = self._get_comp_plot_df(
            comparison_dict
        )
        print(plot_df)

        self._plot(
            plot_df,
            "comparison",
            is_comparison=True
        )

    @overrides
    def visualize(
        self,
        visualizer_params: Dict,
        ds: AllennlpDataset,
    ):
        self.standard_visualize(
            visualizer_params,
            ds
        )

        if visualizer_params["sim_comparison"]["is_sim_comparison"] is True:
            self.comparison_visualize(
                visualizer_params,
                ds
            )


class TSNEVisualizer(visualizer):
    def __init__(
        self,
        visualizer_params: Dict,
        embedder: TextEmbedder,
        encoder: TextEncoder,
        vocab: Vocabulary,
    ):
        self.embedder = embedder
        self.encoder = encoder
        self.vocab = vocab
        self.col_names = visualizer_params["col_names"]
        self.reducers = [
            TSNE(
                n_components=visualizer_params["n_components"],
                perplexity=perplexity,
                n_iter=visualizer_params["n_iter"],
                verbose=visualizer_params["verbose"],
                init="pca"
            ) for perplexity in visualizer_params["perplexity"]
        ]
        self.save_fig_path = visualizer_params["save_fig_path"] + "fig"
        self.GPU = next(encoder.parameters()).get_device()

    @overrides
    def _reduce_dim(
        self,
        reducer: TSNE,
        encode_X: torch.Tensor
    ):
        return reducer.fit_transform(
            encode_X
        )

    @overrides
    def _encode(
        self,
        token_X
    ):
        with torch.no_grad():
            embed_X = self.embedder(token_X)
            tokens_mask = get_text_field_mask(token_X)
            encode_X = self.encoder(embed_X.detach(), tokens_mask)

        return encode_X

    def _get_plot_df(
        self,
        total_X: np.ndarray,
        origin_Y: np.ndarray,
        augment_Y: np.ndarray,
    ):
        origin_Y = np.array(
            list(
                map(
                    lambda x: self.vocab.get_token_from_index(
                        x,
                        namespace="labels"
                    ),
                    list(origin_Y)
                )
            )
        )
        augment_Y = np.array(
            list(
                map(
                    lambda x: self.vocab.get_token_from_index(
                        x,
                        namespace="labels"
                    ) + "_aug",
                    list(augment_Y)
                )
            )
        )

        # Union
        union_Y = np.concatenate([origin_Y, augment_Y])
        union_Y = union_Y.reshape([union_Y.shape[0], 1])
        del origin_Y, augment_Y
        union_arr = np.hstack([total_X, union_Y])
        del total_X, union_Y

        # Get Df
        plot_df = pd.DataFrame(
            union_arr,
            columns=self.col_names
        ).astype({
            self.col_names[0]: "float32",
            self.col_names[1]: "float32",
            self.col_names[2]: "category"
        })
        del union_arr

        return plot_df

    def _visualize(
        self,
        total_X: np.ndarray,
        origin_Y: np.ndarray,
        augment_Y: np.ndarray,
        reducer: TSNE
    ):
        plot_df = self._get_plot_df(
            total_X,
            origin_Y,
            augment_Y,
        )

        palette = sns.color_palette("Paired")
        palette = [
            palette[1],
            palette[0],
            palette[5],
            palette[4]
        ]

        sns.scatterplot(
            data=plot_df,
            x=self.col_names[0],
            y=self.col_names[1],
            size=self.col_names[2],
            size_order=list(plot_df[self.col_names[2]].unique()).sort(),
            sizes=[30, 15, 30, 15],
            hue=self.col_names[2],
            hue_order=list(plot_df[self.col_names[2]].unique()).sort(),
            style=self.col_names[2],
            style_order=list(plot_df[self.col_names[2]].unique()).sort(),
            legend="full",
            palette=palette,
            alpha=1,
            s=30
        )

        plt.savefig(
            self.save_fig_path + '_' + str(reducer.perplexity) + '_' + str(reducer.n_iter),
            dpi=1200
        )
        plt.close()

    def _get_sample_array(
        self,
        total_X: np.ndarray,
        total_Y: np.ndarray,
        total_Z: np.ndarray,
        proportion: float
    ):
        # total_Z responsible for indexing, 1 means the original data
        total_Z = total_Z == 1

        origin_X = total_X[total_Z]
        origin_Y = total_Y[total_Z]
        augment_X = total_X[~total_Z]
        augment_Y = total_Y[~total_Z]

        origin_idx = np.random.choice(
            origin_X.shape[0],
            int(origin_X.shape[0] * proportion),
            replace=False
        )

        augment_idx = np.random.choice(
            augment_X.shape[0],
            int(augment_X.shape[0] * proportion),
            replace=False
        )

        return origin_X[origin_idx, :], origin_Y[origin_idx], augment_X[augment_idx, :], augment_Y[augment_idx]

    def _tsne_visualize(
        self,
        visualizer_params: Dict,
        dataloader: DataLoader,
        reducer: TSNE
    ):
        total_X = []
        total_Y = []
        total_Z = []

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if self.GPU > -1:
                batch = move_to_device(batch, self.GPU)
            else:
                pass

            token_X = batch["token_ids"]
            label_Y = batch["label"].detach().cpu().numpy()
            augment_Z = np.array(batch["augment"])

            encode_X = self._encode(token_X).cpu().numpy()

            total_X.append(encode_X)
            total_Y.append(label_Y)
            total_Z.append(augment_Z)

        total_X = np.vstack(total_X)
        total_Y = np.concatenate(total_Y)
        total_Z = np.concatenate(total_Z)

        origin_X, origin_Y, augment_X, augment_Y = self._get_sample_array(
            total_X,
            total_Y,
            total_Z,
            visualizer_params["proportion"]
        )

        total_X = np.vstack([origin_X, augment_X])
        total_X = self._reduce_dim(reducer, total_X)

        self._visualize(
            total_X,
            origin_Y,
            augment_Y,
            reducer
        )

    @overrides
    def visualize(
        self,
        visualizer_params: Dict,
        ds: AllennlpDataset,
    ):
        dataloader = DataLoader(
            ds,
            batch_size=visualizer_params["batch_size"],
            shuffle=False,
            collate_fn=allennlp_collate
        )

        for reducer in tqdm(self.reducers):
            self._tsne_visualize(
                visualizer_params,
                dataloader,
                reducer
            )


def main():
    pass


if __name__ == '__main__':
    main()

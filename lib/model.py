import torch

from pytorch_metric_learning import losses
from .loss import JsdCrossEntropy, EntropyLoss
from typing import Dict, List
from overrides import overrides
from transformers import get_linear_schedule_with_warmup
from allennlp.nn.util import get_text_field_mask


class SentimentModel(torch.nn.Module):
    def __init__(
        self,
        sentiment_model_params: Dict,
        dataset_dict: Dict,
        embedder: torch.nn.Module,
        encoder: torch.nn.ModuleList,
        classifier: torch.nn.ModuleList
    ):
        super(SentimentModel, self).__init__()

        # Small Module
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier

        # Loss initiailization
        self.classification_criterion = sentiment_model_params["criterions"]["classification_criterion"]
        self.contrastive_criterion = losses.SupConLoss()
        self.consistency_criterion = JsdCrossEntropy()
        self.entropy_criterion = EntropyLoss()

        # Optimizer initialization
        self.optimizer = sentiment_model_params["optimizer"]["select_optimizer"](
            self.parameters(),
            lr=sentiment_model_params["optimizer"]["lr"]
        )

        # Scheduler inititailzation
        if sentiment_model_params["scheduler"]["select_scheduler"] != "none":
            # self.scheduler = sentiment_model_params["scheduler"]["select_scheduler"](self.optimizer)
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=sentiment_model_params["scheduler"]["warmup_steps"],
                num_training_steps=sentiment_model_params["scheduler"]["training_steps"]
            )
        else:
            self.scheduler = None

        # Evaluate initialization
        self.accuracy = sentiment_model_params["evaluation"]

        # Clip initialization
        self.is_clip = sentiment_model_params["clip_grad"]["is_clip"]
        self.max_norm = sentiment_model_params["clip_grad"]["max_norm"]
        self.norm_type = sentiment_model_params["clip_grad"]["norm_type"]

        self.augment_field_names = None

    def set_augment_field_names(
        self,
        augment_field_names: List[str]
    ):
        self.augment_field_names = augment_field_names

    def _get_encoded_X(
        self,
        token_X
    ):
        # Embedded first
        embed_X = self.embedder(token_X)

        # Get token mask for speed up
        tokens_mask = get_text_field_mask(token_X)

        # Encode
        encode_X = self.encoder(embed_X, tokens_mask)

        return encode_X

    def _get_classification_loss_and_predicts(
        self,
        encode_X: torch.Tensor,
        label_Y: torch.Tensor
    ) -> Dict:
        # Classfiy
        pred_Y = self.classifier(encode_X)

        # Get all the loss
        classification_loss = self.classification_criterion(
            pred_Y,
            label_Y
        )

        return classification_loss, pred_Y

    def _standard_forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict:
        output_dict = {}

        # Get input from dict
        encode_X = self._get_encoded_X(batch["text"])
        label_Y = batch["label"]

        # Get Cross entropy Loss
        classification_loss, predicts = self._get_classification_loss_and_predicts(
            encode_X,
            label_Y
        )

        output_dict["classification_loss"] = classification_loss
        output_dict["predicts"] = torch.argmax(predicts, dim=1)

        return output_dict

    def _finetune_forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict:
        output_dict = {}

        # Get input from dict
        origin_encode_X = self._get_encoded_X(batch["text"])
        origin_label_Y = batch["label"]

        # Get Origin Cross entropy Loss
        origin_classification_loss, origin_predicts = self._get_classification_loss_and_predicts(
            origin_encode_X,
            origin_label_Y
        )

        # Get Consistency Loss and Augmented Cross entropy Loss
        assert self.augment_field_names is not None, "Augmented field names is not given!"

        total_augment_loss = 0
        total_consistency_loss = 0

        total_encode_augment_X = self._get_encoded_X(batch[self.augment_field_names[0]])
        total_augment_Y = origin_label_Y
        total_origin_predicts = origin_predicts

        for augment_field_name in self.augment_field_names[1:]:
            new_encode_augment_X = self._get_encoded_X(batch[augment_field_name])
            total_encode_augment_X = torch.cat([
                total_encode_augment_X,
                new_encode_augment_X
            ])
            total_augment_Y = torch.cat([total_augment_Y, origin_label_Y])
            total_origin_predicts = torch.cat([total_origin_predicts, origin_predicts])

        total_augment_loss, total_augment_predicts = self._get_classification_loss_and_predicts(
            total_encode_augment_X,
            total_augment_Y
        )
        total_consistency_loss = self.consistency_criterion(
            total_origin_predicts,
            total_augment_predicts
        )
        total_sc_loss = self.contrastive_criterion(
            torch.cat([origin_encode_X, total_encode_augment_X]),
            torch.cat([origin_label_Y, total_augment_Y])
        )

        total_entropy_loss = self.entropy_criterion(
            torch.cat([origin_predicts, total_augment_predicts])
        )

        output_dict["origin_classification_loss"] = origin_classification_loss
        output_dict["total_augment_loss"] = total_augment_loss
        output_dict["total_consistency_loss"] = total_consistency_loss
        output_dict["total_sc_loss"] = total_sc_loss
        output_dict["total_entropy_loss"] = total_entropy_loss
        output_dict["predicts"] = torch.argmax(origin_predicts, dim=1)

        return output_dict

    @overrides
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        is_finetune: bool
    ) -> Dict:
        if is_finetune is False:
            output_dict = self._standard_forward(batch)
        else:
            output_dict = self._finetune_forward(batch)

        return output_dict

    def optimize(
        self,
        loss,
        optimizers,
        is_step: bool = True,
    ):
        loss.backward()

        for optimizer in optimizers:
            if self.is_clip is True:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.max_norm,
                    self.norm_type
                )
            else:
                pass

            if is_step is True:
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                else:
                    pass
                optimizer.zero_grad()
            else:
                pass


def main():
    pass


if __name__ == '__main__':
    main()

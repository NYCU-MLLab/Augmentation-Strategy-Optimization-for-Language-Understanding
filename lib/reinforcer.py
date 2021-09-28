import torch
import random
import numpy as np

from overrides import overrides
from typing import List, Dict
from .loss import JsdCrossEntropy, IBLoss, EntropyLoss
# from .embedder import SentenceEmbedder
from .utils import get_sentence_from_text_field_tensors
from .augmenter import Augmenter
from allennlp.nn.util import get_token_ids_from_text_field_tensors, get_text_field_mask
from allennlp.modules.feedforward import FeedForward


class Environment(torch.nn.Module):
    def __init__(
        self,
        env_params: Dict,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        # SE_embedder: SentenceEmbedder,
        augmenter_list: List[Augmenter]
    ):
        super(Environment, self).__init__()
        # Module initialization
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier
        self.augmenter_list = augmenter_list
        # self.SE_embedder = SE_embedder

        # Environment Variable
        self.initial_state = None
        self.initial_encoded_state = None
        self.initial_prediction = None
        self.previous_state = None
        self.previous_reward = 0
        self.label = None

        # Calculation Function
        self.lambda_multiplier = 1.2
        self.entropy_multiplier = 0.9
        self.maximize_target_func_name = env_params["maximize_target"]
        self.loss_reward = self._get_maximize_target_func(self.maximize_target_func_name)
        self.distance_penalty = torch.nn.MSELoss()
        self.entropy_reward = EntropyLoss()
        self.cos_similarity = torch.nn.CosineSimilarity()
        self.similarity_threshold = env_params["similarity_threshold"]

        # Record Variable
        self.failed_num = 0
        self.sim = [0, 0, 0, 0]
        self.sim_action = [0, 0, 0, 0]

    def _get_maximize_target_func(
        self,
        select_func_name: str
    ):
        if select_func_name == "cross-entropy":
            return torch.nn.CrossEntropyLoss()
        elif select_func_name == "consistency":
            return JsdCrossEntropy()
        elif select_func_name == "ib":
            return IBLoss()
        else:
            raise ValueError("Not Support Error")

    def get_encoded_state(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        training_status = self.training

        self.eval()
        with torch.no_grad():
            # Embedded first
            embedded_state = self.embedder(state)

            # get token mask
            state_mask = get_text_field_mask(state)

            # Encode
            encoded_state = self.encoder(embedded_state, state_mask)

        if training_status is True:
            self.train()
        else:
            pass

        return encoded_state

    def _get_initial_reward(
        self,
        initial_prediction: torch.Tensor,
        label: torch.Tensor
    ):
        if self.maximize_target_func_name == "consistency":
            return 0
        elif self.maximize_target_func_name == "cross-entropy":
            return self.loss_reward(initial_prediction, label).detach().cpu().item()
        elif self.maximize_target_func_name == "ib":
            return self.loss_reward(initial_prediction, label).detach().cpu().item()
        else:
            raise ValueError("Not surport!")

    def reset(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]],
        label: torch.Tensor
    ):
        self.initial_state = wrapped_token_of_sent
        # self.initial_SE = self.SE_embedder(wrapped_token_of_sent)
        self.previous_state = wrapped_token_of_sent
        self.label = label
        self.initial_encoded_state = self.get_encoded_state(self.initial_state)

        training_status = self.training
        self.eval()
        with torch.no_grad():
            self.initial_prediction = self.classifier(self.initial_encoded_state)
        if training_status is True:
            self.train
        else:
            pass

        self.previous_reward = self._get_initial_reward(self.initial_prediction, self.label)

        return self.previous_state

    def _get_loss_reward(
        self,
        initial_prediction: torch.Tensor,
        augmented_prediction: torch.Tensor,
        label: torch.Tensor
    ):
        if self.maximize_target_func_name == "consistency":
            return self.loss_reward(initial_prediction, augmented_prediction).detach().cpu().item()
        elif self.maximize_target_func_name == "cross-entropy":
            return self.loss_reward(augmented_prediction, label).detach().cpu().item()
        elif self.maximize_target_func_name == "ib":
            return self.loss_reward(augmented_prediction, label).detach().cpu().item()
        else:
            raise ValueError("Not surport!")

    def _get_env_respond(
        self,
        augmented_state: Dict[str, Dict[str, torch.Tensor]],
        action_probs: torch.Tensor,
        action: int
    ):
        done = False

        # Get encoded augmented sentence embedding for similarity calculation preparation
        encoded_augmented_state = self.get_encoded_state(augmented_state)

        # Calculate Reward - Typical
        training_status = self.training
        self.eval()
        with torch.no_grad():
            augmented_prediction = self.classifier(encoded_augmented_state)
        if training_status is True:
            self.train()
        else:
            pass

        # Entropy
        entropy_reward = self.entropy_reward(action_probs).detach().cpu().item()

        # Loss
        loss_reward = self._get_loss_reward(self.initial_prediction, augmented_prediction, self.label)

        # Distance
        distance_penalty = self.lambda_multiplier * self.distance_penalty(
            self.initial_encoded_state,
            encoded_augmented_state
        ).detach().cpu().item()

        # Similarity threshold
        if self.cos_similarity(self.initial_encoded_state, encoded_augmented_state).item() < self.similarity_threshold:
            self.failed_num += 1
            loss_reward = 0
            entropy_reward = 0
            done = True
        else:
            # Move to next state
            self.previous_state = augmented_state

        self.sim[action] += self.cos_similarity(self.initial_encoded_state, encoded_augmented_state).item()
        self.sim_action[action] += 1

        # Final
        final_reward = loss_reward - distance_penalty - self.previous_reward + entropy_reward * self.entropy_multiplier

        self.previous_reward = final_reward

        return done, final_reward

    def step(
        self,
        action: int,
        action_probs: torch.Tensor
    ):
        done = False
        reward = 0.0

        # Last action will be "stop"
        if action == len(self.augmenter_list) - 1:
            done = True
        else:
            augmented_state = self.augmenter_list[action].augment(self.previous_state)
            done, reward = self._get_env_respond(augmented_state, action_probs, action)

        return self.previous_state, reward, done


class Policy(torch.nn.Module):
    def __init__(
        self,
        policy_params: Dict,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        input_dim: int,
        num_of_action: int
    ):
        super(Policy, self).__init__()

        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier

        self.feedforward = FeedForward(
            input_dim,
            num_layers=policy_params["feedforward"]["num_layers"]-1,
            hidden_dims=policy_params["feedforward"]["hidden_dims"][:-1],
            activations=policy_params["feedforward"]["activations"],
            dropout=policy_params["feedforward"]["dropout"]
        )
        self.final_feedforward = torch.nn.Linear(
            self.feedforward.get_output_dim(),
            policy_params["feedforward"]["hidden_dims"][-1]
        )

        self.optimizer = policy_params["optimizer"]["select_optimizer"](
            self.parameters(),
            lr=policy_params["optimizer"]["lr"]
        )

        self.initial_encoded_state = None
        self.initial_prediction = None

    def reset(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        self.initial_encoded_state = self.get_encoded_state(state)

        training_status = self.training
        self.eval()
        with torch.no_grad():
            self.initial_prediction = self.classifier(self.initial_encoded_state)
        if training_status is True:
            self.train()
        else:
            pass

    def select_action(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        action, action_prob, action_probs = self(state)

        return action, action_prob, action_probs

    def get_encoded_state(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        training_status = self.training

        self.eval()
        with torch.no_grad():
            # Embedded first
            embedded_state = self.embedder(state)

            # get token mask
            state_mask = get_text_field_mask(state)

            # Encode
            encoded_state = self.encoder(embedded_state, state_mask)
        if training_status is True:
            self.train()
        else:
            pass

        return encoded_state

    @overrides
    def forward(
        self,
        state: Dict[str, Dict[str, torch.Tensor]]
    ):
        encoded_state = self.get_encoded_state(state)
        real_state = torch.cat((self.initial_prediction, encoded_state), 1)

        # Get action probs
        action_scores = self.final_feedforward(self.feedforward(real_state.detach()))
        action_probs = torch.nn.functional.softmax(action_scores, dim=1)

        # Get action
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        return action.item(), m.log_prob(action), action_scores


class REINFORCER(torch.nn.Module):
    def __init__(
        self,
        env_params: Dict,
        policy_params: Dict,
        REINFORCE_params: Dict,
        dataset_dict: Dict,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        augmenter_list: List[Augmenter]
    ):
        super(REINFORCER, self).__init__()

        embedder.eval()
        encoder.eval()
        classifier.eval()

        self.policy = Policy(
            policy_params,
            embedder,
            encoder,
            classifier,
            encoder.get_output_dim() + classifier.get_output_dim(),
            len(augmenter_list)
        )
        self.env = Environment(
            env_params,
            embedder,
            encoder,
            classifier,
            # UniversalSentenceEmbedder(dataset_dict["dataset_reader"].transformer_tokenizer),
            augmenter_list
        )

        self.dataset_vocab = dataset_dict["dataset_vocab"]
        self.transformer_tokenizer = dataset_dict["dataset_reader"].transformer_tokenizer
        self.transformer_vocab = dataset_dict["dataset_reader"].transformer_vocab

        self.max_step = REINFORCE_params["max_step"]
        self.gamma = REINFORCE_params["gamma"]
        self.clip_grad = REINFORCE_params["clip_grad"]

    def _get_token_of_sents(
        self,
        token_ids: Dict[str, Dict[str, torch.Tensor]]
    ):
        token_of_sents = get_token_ids_from_text_field_tensors(token_ids)

        return token_of_sents

    def _wrap_token_of_sent(
        self,
        token_of_sent: torch.Tensor
    ):
        wrapped_token_of_sent = torch.stack([token_of_sent])

        return {"tokens": {"token_ids": wrapped_token_of_sent}}

    def _calculate_loss(
        self,
        log_probs,
        rewards
    ):
        R = 0.0
        losses = []
        returns = []

        # Calculate cumulated reward
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # If the length equals to one, we do not need to do any standardlization
        if len(rewards) == 1:
            pass
        else:
            returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        # Calculate loss
        for log_prob, R in zip(log_probs, returns):
            losses.append(-log_prob * R)

        loss = torch.cat(losses).sum()

        return loss

    def optimize(
        self,
        loss
    ):
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)

        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()

    def EDA_augment(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]],
        mode_params: Dict
    ):
        augmented_state = wrapped_token_of_sent
        for stack_step in range(mode_params["stack_step"]):
            action = random.choice(range(len(self.env.augmenter_list) - 1))
            augmented_state = self.env.augmenter_list[action].augment(augmented_state)

        return get_sentence_from_text_field_tensors(
            self.transformer_tokenizer,
            augmented_state,
        )

    def BackTrans_augment(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        augmented_state = self.env.augmenter_list[0].augment(wrapped_token_of_sent)

        return get_sentence_from_text_field_tensors(
            self.transformer_tokenizer,
            augmented_state
        )

    def augment(
        self,
        episode
    ):
        state = self.env.reset(episode["text"], episode["label"])
        self.policy.reset(episode["text"])
        log_probs = []
        rewards = []
        actions = []

        for step in range(self.max_step):
            action, action_log_prob, action_probs = self.policy.select_action(state)
            state, reward, done = self.env.step(action, action_probs)
            actions.append(action)

            log_probs.append(action_log_prob)
            rewards.append(reward)

            if done is True:
                break

        return get_sentence_from_text_field_tensors(
            self.transformer_tokenizer,
            self.env.previous_state
        )

    @overrides
    def forward(
        self,
        episode: Dict[str, torch.Tensor]
    ):
        """
        forward: get loss(output dict) from currenet episode(currenet sentence)
        """
        output_dict = {}

        wrapped_token_of_sent = episode["text"]
        label = episode["label"]

        state = self.env.reset(wrapped_token_of_sent, label)
        self.policy.reset(wrapped_token_of_sent)
        log_probs = []
        rewards = []
        actions = []

        for step in range(self.max_step):
            action, action_log_prob, action_probs = self.policy.select_action(state)
            state, reward, done = self.env.step(action, action_probs)

            log_probs.append(action_log_prob)
            rewards.append(reward)
            actions.append(action)

            if done is True:
                break

        # calculate loss
        loss = self._calculate_loss(log_probs, rewards)

        # Prepare output dict
        output_dict["origin_sentence"] = get_sentence_from_text_field_tensors(
            self.transformer_tokenizer,
            wrapped_token_of_sent
        )
        output_dict["augment_sentence"] = get_sentence_from_text_field_tensors(
            self.transformer_tokenizer,
            self.env.previous_state
        )
        output_dict["actions"] = actions
        output_dict["loss"] = loss
        output_dict["ep_reward"] = torch.sum(torch.tensor(rewards), dim=0).detach().cpu().item()

        return output_dict


def main():
    pass


if __name__ == '__main__':
    main()

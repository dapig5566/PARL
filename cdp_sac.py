from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
import torch
import tree  # pip install dm_tree
from typing import Dict, List, Optional
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType


class CDPSACTorchModel(SACTorchModel):
    def _get_q_value(self, model_out, actions, net):
        # Model outs may come as original Tuple observations, concat them
        # here if this is the case.
        # if isinstance(net.obs_space, Box):
        #     if isinstance(model_out, (list, tuple)):
        #         model_out = torch.cat(model_out, dim=-1)
        #     elif isinstance(model_out, dict):
        #         model_out = torch.cat(list(model_out.values()), dim=-1)

        # Continuous case -> concat actions to model_out.
        if actions is not None:
            if self.concat_obs_and_actions:
                input_dict = {"obs": torch.cat([model_out, actions], dim=-1)}
            else:
                # TODO(junogng) : SampleBatch doesn't support list columns yet.
                #     Use ModelInputDict.
                input_dict = {"obs": (model_out, actions)}
        # Discrete case -> return q-vals for all actions.
        else:
            input_dict = {"obs": model_out}
        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        input_dict["is_training"] = True

        return net(input_dict, [], None)

    def get_action_model_outputs(
        self,
        model_out: TensorType,
        state_in: List[TensorType] = None,
        seq_lens: TensorType = None,
    ) -> (TensorType, List[TensorType]):
        """Returns distribution inputs and states given the output of
        policy.model().

        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.

        Args:
            model_out: Feature outputs from the model layers
                (result of doing `model(obs)`).
            state_in List(TensorType): State input for recurrent cells
            seq_lens: Sequence lengths of input- and state
                sequences

        Returns:
            TensorType: Distribution inputs for sampling actions.
        """

        def concat_obs_if_necessary(obs: TensorStructType):
            """Concat model outs if they come as original tuple observations."""
            if isinstance(obs, (list, tuple)):
                obs = torch.cat(obs, dim=-1)
            elif isinstance(obs, dict):
                obs = torch.cat(
                    [
                        torch.unsqueeze(val, 1) if len(val.shape) == 1 else val
                        for val in tree.flatten(obs.values())
                    ],
                    dim=-1,
                )
            return obs

        if state_in is None:
            state_in = []

        # if isinstance(model_out, dict) and "obs" in model_out:
        #     # Model outs may come as original Tuple observations
        #     if isinstance(self.action_model.obs_space, Box):
        #         model_out["obs"] = concat_obs_if_necessary(model_out["obs"])
        #     return self.action_model(model_out, state_in, seq_lens)
        # else:
        #     if isinstance(self.action_model.obs_space, Box):
        #         model_out = concat_obs_if_necessary(model_out)
        return self.action_model({"obs": model_out}, state_in, seq_lens)

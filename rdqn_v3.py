
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import LSTMWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel
from transformer import TransformerModel, generate_square_subsequent_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from config import USER_INDEX, NUM_CARS, NUM_COURIERS, NUM_ACTIONS, num_hist

USER_EMBEDDING_SIZE = 256
TRANSFORMER_HIDDEN = 1024
N_ENCODER_LAYERS = 1
N_DECODER_LAYERS = 3
N_HEAD = 8
STATE_SIZE = 128
FUSION_SIZE = 128
TSF_FUSION_SIZE = 256
DNN_HIDDEN = 64
CNN_HIDDEN = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1


class CDPModel_RDQN_v3(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, device=None, ablation=None, **kwargs):
        num_outputs = USER_EMBEDDING_SIZE
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.ablation = ablation
        self._device = device
        
        # print(model_config)
        # print(kwargs)

        # custom_model_config = model_config.pop("custom_model_config", {}) if kwargs == {} else kwargs.pop("custom_model_config", {})
        self.num_cars = action_space.n - 1
        self.num_couriers = 1
        
        # self.num_cars = 4
        # self.num_couriers = 1

        # print("model: {}".format(self.num_cars))
        # print("model: {}".format(self.num_couriers))

        # pre-train
        self.user_embedding = nn.Embedding(len(USER_INDEX) + 1 + 5, USER_EMBEDDING_SIZE, padding_idx=0)
        self.short_term_token = torch.tensor([len(USER_INDEX) + 1]).long()
        self.current_order_token = torch.tensor([len(USER_INDEX) + 2]).long()
        self.dist_query_token = torch.tensor([len(USER_INDEX) + 3]).long()
        self.can_predict_token = torch.tensor([len(USER_INDEX) + 4]).long()
        self.cannot_predict_token = torch.tensor([len(USER_INDEX) + 5]).long()

        # pre-train
        

        self.encoder = nn.Sequential(
            nn.Linear(in_features=29 + 7 + 1 + USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
        )

        self.map_encoder3 = nn.Sequential(
            nn.Conv2d(
                in_channels=7 * self.num_cars + self.num_couriers,
                out_channels=CNN_HIDDEN,
                kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                stride=(STRIDE, STRIDE),
                padding=PADDING,
                bias=False
            ),
            nn.BatchNorm2d(num_features=CNN_HIDDEN),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=CNN_HIDDEN,
                out_channels=CNN_HIDDEN,
                kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                stride=(STRIDE, STRIDE),
                padding=PADDING,
                bias=False
            ),
            nn.BatchNorm2d(num_features=CNN_HIDDEN),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=CNN_HIDDEN,
                out_channels=CNN_HIDDEN,
                kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                stride=(STRIDE, STRIDE),
                padding=PADDING,
                bias=False
            ),
            nn.BatchNorm2d(num_features=CNN_HIDDEN),
            nn.PReLU(),
        )
        self.cnn_pool = nn.AdaptiveAvgPool2d(output_size=1)
       
        self.fusion3 = nn.Sequential(
            nn.Linear(in_features=USER_EMBEDDING_SIZE + TSF_FUSION_SIZE + CNN_HIDDEN,
                    out_features=TSF_FUSION_SIZE),
            nn.BatchNorm1d(TSF_FUSION_SIZE),
            nn.ReLU()
        )
        self.to_q = nn.Sequential(
                nn.Linear(in_features=USER_EMBEDDING_SIZE,
                        out_features=TSF_FUSION_SIZE),
                nn.ReLU()
            )
        self.to_k = nn.Sequential(
                nn.Linear(in_features=CNN_HIDDEN,
                        out_features=TSF_FUSION_SIZE),
                nn.ReLU()
            )
        self.to_v = nn.Sequential(
                nn.Linear(in_features=CNN_HIDDEN,
                        out_features=TSF_FUSION_SIZE),
                nn.ReLU()
            )

    def forward(self, input_dict, state, seq_lens):

        current_input = input_dict["obs"][0]
        order = current_input[0]
        pred_time = current_input[1]
        entropy = current_input[2]
        uid = current_input[2]
        
        with torch.no_grad():
            uid = uid.long()
            user_init = self.user_embedding(uid)

        # print(user_init.size(), order.size(), pred_time.size(), entropy.size())
        mixed_emb = torch.cat([user_init, order, pred_time, entropy.unsqueeze(1)], dim=-1)
        mixed_repr = self.encoder(mixed_emb)
        
        # process the regions' heat map (in Ablation)
        dispatch_map = input_dict["obs"][1]
        dispatch_map_repr = self.map_encoder3(dispatch_map)

        # print(dispatch_map_repr.size())
        dispatch_map_req = dispatch_map_repr.view(dispatch_map_repr.size(0), CNN_HIDDEN, -1)
        dispatch_map_req = dispatch_map_req.permute(0, 2, 1).contiguous()
        mixed_q = self.to_q(mixed_repr)
        map_k = self.to_k(dispatch_map_req)
        map_v = self.to_v(dispatch_map_req)

        attn_score = torch.softmax(torch.einsum("bd, bsd->bs", mixed_q, map_k), dim=1)
        attn_out = torch.einsum("bs, bsd->bd", attn_score, map_v)
        
        dispatch_map_repr = self.cnn_pool(dispatch_map_repr).squeeze()
        if len(dispatch_map_repr.size()) == 1:
            dispatch_map_repr = dispatch_map_repr.unsqueeze(0)
        
        cat_feature = torch.cat([mixed_repr, attn_out, dispatch_map_repr], dim=-1)
        self.inner_feature = self.fusion3(cat_feature)

        return self.inner_feature, state
    
        
class RecurrentDQN(LSTMWrapper, CDPModel_RDQN_v3):
    pass

RecurrentDQN._wrapped_forward = CDPModel_RDQN_v3.forward

class WrappedRecurrentDQN(DQNTorchModel, RecurrentDQN):
    pass



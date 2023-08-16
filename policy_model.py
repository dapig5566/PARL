from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import LSTMWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel
from transformer import TransformerModel, generate_square_subsequent_mask
import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from config import USER_INDEX, NUM_CARS, NUM_COURIERS, NUM_ACTIONS, num_hist

USER_EMBEDDING_SIZE = 256
TRANSFORMER_HIDDEN = 1024
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3
N_HEAD = 8
STATE_SIZE = 128
FUSION_SIZE = 128
TSF_FUSION_SIZE = 256
DNN_HIDDEN = 64
CNN_HIDDEN = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1


class CDPModel_DQN_with_pretrain(DQNTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, pre_train=False, **kwargs):
        num_outputs = FUSION_SIZE
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        self.pre_train = pre_train

        # pre-train
        self.user_embedding = nn.Embedding(len(USER_INDEX) + 1, USER_EMBEDDING_SIZE, padding_idx=0)

        self.state_emb = nn.Sequential(
            nn.Linear(in_features=USER_EMBEDDING_SIZE + 29 + NUM_ACTIONS,
                      out_features=STATE_SIZE),
            nn.BatchNorm1d(STATE_SIZE),
            nn.ReLU(),
        )
        self.state_encoder = nn.LSTM(USER_EMBEDDING_SIZE + 29 + NUM_ACTIONS, STATE_SIZE, batch_first=True)

        self.remained_emb = nn.Sequential(
            nn.Linear(in_features=USER_EMBEDDING_SIZE + 29,
                      out_features=STATE_SIZE),
            nn.ReLU(),
        )
        self.remained_encoder = nn.LSTM(USER_EMBEDDING_SIZE + 29, STATE_SIZE, batch_first=True)

        # pre-train
        self.hist_emb = nn.Sequential(
            nn.Linear(in_features=36,
                      out_features=USER_EMBEDDING_SIZE // 2),
            nn.ReLU(),
        )

        # pre-train
        self.hist_encoder = nn.LSTM(USER_EMBEDDING_SIZE // 2, USER_EMBEDDING_SIZE // 2, batch_first=True)


        self.order_encoder = nn.Sequential(
            nn.Linear(in_features=29,
                      out_features=USER_EMBEDDING_SIZE // 2),
            nn.BatchNorm1d(USER_EMBEDDING_SIZE // 2),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=7 * NUM_CARS + NUM_COURIERS,
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
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        # self.fusion = nn.Sequential(
        #     nn.Linear(in_features=STATE_SIZE + USER_EMBEDDING_SIZE // 2 + STATE_SIZE + CNN_HIDDEN,
        #               out_features=FUSION_SIZE),
        #     nn.ReLU()
        # )

        # pre-train
        self.norm = nn.LayerNorm([USER_EMBEDDING_SIZE // 2])
        self.att_coeff = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        self.level_estimator = nn.Sequential(
            nn.Linear(in_features=num_hist,
                      out_features=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.fusion3 = nn.Sequential(
            nn.Linear(in_features=USER_EMBEDDING_SIZE // 2 + 1,
                      out_features=FUSION_SIZE),
            nn.BatchNorm1d(FUSION_SIZE),
            nn.ReLU()
        )

    def forward(self, input_dict, state, seq_lens):
        # process dispatch state
        # current_state = input_dict["obs"][0]
        # state_orders = current_state[0]
        # state_uids = current_state[1]
        #
        # with torch.no_grad():
        #     user_repr = self.user_embedding(state_uids.int())
        #     state_orders = torch.cat([user_repr, state_orders], dim=-1)
        #
        # state_repr, _ = self.state_encoder(state_orders)
        #
        # process current order to dispatch
        current_input = input_dict["obs"][0]
        order = current_input[0]
        hist = current_input[1]
        uid = current_input[2]
        # print(order.shape)
        # print(hist.shape)
        # print(uid.shape)
        # assert False
        order_repr = self.order_encoder(order)
        if not self.pre_train:
            uid = uid.long()
        user_init = self.user_embedding(uid)
        ih, ic = torch.split(user_init, USER_EMBEDDING_SIZE // 2, dim=-1)
        embedded_hist = self.hist_emb(hist).contiguous()
        hist_repr, _ = self.hist_encoder(embedded_hist, (ih.unsqueeze(0).contiguous(), ic.unsqueeze(0).contiguous()))

        # attentive user representation (B, L, D)

        attention_logit = torch.einsum("bld,bdk->blk", hist_repr, order_repr.unsqueeze(2))
        attention_repr = torch.sum(torch.softmax(attention_logit, dim=1) * hist_repr, dim=1)

        level_est = self.level_estimator(attention_logit.squeeze(2))

        attention_repr = self.norm(self.att_coeff * attention_repr + order_repr)


        # process remained orders haven't been dispatched
        # remained_state = input_dict["obs"][2]
        # remained_orders = remained_state[0]
        # remained_uids = remained_state[1]
        #
        # with torch.no_grad():
        #     remained_user_repr = self.user_embedding(remained_uids.int())
        #     remained_orders = torch.cat([remained_user_repr, remained_orders], dim=-1)
        #
        # remained_repr, _ = self.remained_encoder(remained_orders)
        # #

        # # process the regions' heat map (in Ablation)
        # dispatch_map = input_dict["obs"][1]
        # dispatch_map_repr = self.map_encoder(dispatch_map).squeeze()
        # if len(dispatch_map_repr.shape) == 1:
        #     dispatch_map_repr = dispatch_map_repr.unsqueeze(0)

        # cat_feature = torch.cat([state_repr[:, -1, :],
        #                          order_repr,
        #                          hist_repr[:, -1, :],
        #                          remained_repr[:, -1, :],
        #                          dispatch_map_repr], dim=-1)

        # cat_feature = torch.cat([state_repr[:, -1, :],
        #                          attention_repr,
        #                          remained_repr[:, -1, :],
        #                          dispatch_map_repr], dim=-1)

        # cat_feature = torch.cat([attention_repr, level_est, dispatch_map_repr], dim=-1)

        # dispatch map ablation
        cat_feature = torch.cat([attention_repr, level_est], dim=-1)

        self.inner_feature = self.fusion3(cat_feature)

        return self.inner_feature, state

class CDPModel_AlphaZero(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ActorCriticModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        
        self.pre_train = False
        self._device = None
        self.recurrent = False
        # pre-train
        self.user_embedding = nn.Embedding(len(USER_INDEX) + 1 + 5, USER_EMBEDDING_SIZE, padding_idx=0)
        self.short_term_token = torch.tensor([len(USER_INDEX) + 1]).long()
        self.current_order_token = torch.tensor([len(USER_INDEX) + 2]).long()
        self.dist_query_token = torch.tensor([len(USER_INDEX) + 3]).long()
        self.can_predict_token = torch.tensor([len(USER_INDEX) + 4]).long()
        self.cannot_predict_token = torch.tensor([len(USER_INDEX) + 5]).long()

        # pre-train
        self.hist_emb = nn.Sequential(
            nn.Linear(in_features=36,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
        )

        self.order_encoder = nn.Sequential(
            nn.Linear(in_features=29,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
        )

        
        self.transformer = TransformerModel(
            d_model=USER_EMBEDDING_SIZE,
            nhead=N_HEAD,
            d_hid=TRANSFORMER_HIDDEN,
            enc_nlayers=N_ENCODER_LAYERS,
            dec_nlayers=N_DECODER_LAYERS,
            dropout=0.2
        )

        self.map_encoder5 = nn.Sequential(
            nn.Conv2d(
                in_channels=7 * NUM_CARS + NUM_COURIERS,
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
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        if self.recurrent:
            self.state_encoder = nn.LSTM(USER_EMBEDDING_SIZE, USER_EMBEDDING_SIZE, batch_first=True)

        self.fusion3 = nn.Sequential(
            nn.Linear(in_features=USER_EMBEDDING_SIZE + CNN_HIDDEN,
                      out_features=TSF_FUSION_SIZE),
            # nn.BatchNorm1d(TSF_FUSION_SIZE),
            nn.ReLU()
        )
        self.logits = nn.Sequential(
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=TSF_FUSION_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=action_space.n)
        )
        self.vf = nn.Sequential(
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=TSF_FUSION_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=1)
        )

    def forward(self, input_dict, state, seq_lens):
        # process dispatch state
        # current_state = input_dict["obs"][0]
        # state_orders = current_state[0]
        # state_uids = current_state[1]
        #
        # with torch.no_grad():
        #     user_repr = self.user_embedding(state_uids.int())
        #     state_orders = torch.cat([user_repr, state_orders], dim=-1)
        #
        # state_repr, _ = self.state_encoder(state_orders)
        #
        # process current order to dispatch
        current_input = input_dict["obs"][0]
        order = current_input[0]
        hist = current_input[1]
        uid = current_input[2]

        # print(order.shape)
        # print(hist.shape)
        # print(uid.shape)
        # assert False
        order_repr = self.order_encoder(order).unsqueeze(1).contiguous()
        if not self.pre_train:
            uid = uid.long()
        user_init = self.user_embedding(uid).unsqueeze(1)
        embedded_hist = self.hist_emb(hist)
        
        # lt_token_emb = self.user_embedding(self.long_term_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        st_token_emb = self.user_embedding(self.short_term_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        co_token_emb = self.user_embedding(self.current_order_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        dq_token_emb = self.user_embedding(self.dist_query_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1).permute(1, 0, 2)
        # cp_token_emb = self.user_embedding(self.can_predict_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        # cnp_token_emb = self.user_embedding(self.cannot_predict_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        

        long_short_term_user_hist = torch.cat([user_init, st_token_emb, embedded_hist, co_token_emb, order_repr], dim=1).contiguous().permute(1, 0, 2)
        
        
        src_mask = generate_square_subsequent_mask(long_short_term_user_hist.size(0))
        tgt_mask = generate_square_subsequent_mask(dq_token_emb.size(0))
        
        
        src_mask = src_mask.to(long_short_term_user_hist.device)
        tgt_mask = tgt_mask.to(long_short_term_user_hist.device)

        transformer_output = self.transformer(src=long_short_term_user_hist, 
                                  tgt=dq_token_emb, 
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask)
        
        transformer_output = transformer_output.permute(1, 0, 2).squeeze()
        if len(transformer_output.size()) == 1:
            transformer_output = transformer_output.unsqueeze(0)
        
        # # process the regions' heat map (in Ablation)
        dispatch_map = input_dict["obs"][1]
        dispatch_map_repr = self.map_encoder5(dispatch_map).squeeze()
        if len(dispatch_map_repr.size()) == 1:
            dispatch_map_repr = dispatch_map_repr.unsqueeze(0)


        # cat_feature = torch.cat([transformer_output, dispatch_map_repr], dim=-1)

        # dispatch map ablation
        # cat_feature = torch.cat([attention_repr, level_est], dim=-1)

        # self.inner_feature = self.fusion3(cat_feature)
        
        cat_feature = torch.cat([transformer_output, dispatch_map_repr], dim=-1)

        all_feature = self.fusion3(cat_feature)

        self.inner_feature = all_feature

        return self.logits(self.inner_feature), state

    def value_function(self):
        return self.vf(self.inner_feature).squeeze(1)
    


class CDPModel_TDQN_with_pretrain(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, recurrent=False, pre_train=False, device=None, ablation=None, **kwargs):
        num_outputs = USER_EMBEDDING_SIZE
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.ablation = ablation
        self.pre_train = pre_train
        self._device = device
        self.recurrent = recurrent
        # pre-train
        self.user_embedding = nn.Embedding(len(USER_INDEX) + 1 + 5, USER_EMBEDDING_SIZE, padding_idx=0)
        self.short_term_token = torch.tensor([len(USER_INDEX) + 1]).long()
        self.current_order_token = torch.tensor([len(USER_INDEX) + 2]).long()
        self.dist_query_token = torch.tensor([len(USER_INDEX) + 3]).long()
        self.can_predict_token = torch.tensor([len(USER_INDEX) + 4]).long()
        self.cannot_predict_token = torch.tensor([len(USER_INDEX) + 5]).long()

        # pre-train
        self.hist_emb = nn.Sequential(
            nn.Linear(in_features=36,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
        )

        self.order_encoder = nn.Sequential(
            nn.Linear(in_features=29,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=USER_EMBEDDING_SIZE,
                      out_features=USER_EMBEDDING_SIZE),
            nn.ReLU(),
        )

        
        self.transformer = TransformerModel(
            d_model=USER_EMBEDDING_SIZE,
            nhead=N_HEAD,
            d_hid=TRANSFORMER_HIDDEN,
            enc_nlayers=N_ENCODER_LAYERS,
            dec_nlayers=N_DECODER_LAYERS,
            dropout=0.2
        )

        self.map_encoder3 = nn.Sequential(
            nn.Conv2d(
                in_channels=7 * NUM_CARS + NUM_COURIERS,
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
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        if self.recurrent:
            self.state_encoder = nn.LSTM(USER_EMBEDDING_SIZE, USER_EMBEDDING_SIZE, batch_first=True)

        if self.ablation is not None and self.ablation == "user_repr":
            self.user_encoder = nn.Sequential(
            nn.Linear(in_features=num_hist*USER_EMBEDDING_SIZE + USER_EMBEDDING_SIZE,
                      out_features=TSF_FUSION_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=TSF_FUSION_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=TSF_FUSION_SIZE,
                      out_features=TSF_FUSION_SIZE),
            nn.ReLU()
        )
        if not self.pre_train:
            self.fusion3 = nn.Sequential(
                nn.Linear(in_features=USER_EMBEDDING_SIZE + CNN_HIDDEN,
                        out_features=TSF_FUSION_SIZE),
                nn.BatchNorm1d(TSF_FUSION_SIZE),
                nn.ReLU()
            )
        else:
            self.fusion3 = nn.Sequential(
                nn.Linear(in_features=USER_EMBEDDING_SIZE,
                        out_features=TSF_FUSION_SIZE),
                nn.BatchNorm1d(TSF_FUSION_SIZE),
                nn.ReLU()
            )

    def forward(self, input_dict, state, seq_lens):
        # process dispatch state
        # current_state = input_dict["obs"][0]
        # state_orders = current_state[0]
        # state_uids = current_state[1]
        #
        # with torch.no_grad():
        #     user_repr = self.user_embedding(state_uids.int())
        #     state_orders = torch.cat([user_repr, state_orders], dim=-1)
        #
        # state_repr, _ = self.state_encoder(state_orders)
        #
        # process current order to dispatch
        current_input = input_dict["obs"][0]
        order = current_input[0]
        hist = current_input[1]
        uid = current_input[2]

        # print(order.shape)
        # print(hist.shape)
        # print(uid.shape)
        # assert False
        order_repr = self.order_encoder(order).unsqueeze(1).contiguous()
        if not self.pre_train:
            uid = uid.long()
        user_init = self.user_embedding(uid).unsqueeze(1)
        embedded_hist = self.hist_emb(hist)
        
        # lt_token_emb = self.user_embedding(self.long_term_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        st_token_emb = self.user_embedding(self.short_term_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        co_token_emb = self.user_embedding(self.current_order_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        dq_token_emb = self.user_embedding(self.dist_query_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1).permute(1, 0, 2)
        # cp_token_emb = self.user_embedding(self.can_predict_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        # cnp_token_emb = self.user_embedding(self.cannot_predict_token.to(user_init.device)).tile((user_init.size(0), 1)).unsqueeze(1)
        
        if self.ablation is not None and self.ablation == "user_repr":
            bs = order_repr.size(0)
            raw_input = torch.cat([order_repr, embedded_hist], dim=1).view(bs, -1)
            user_repr = self.user_encoder(raw_input)
            transformer_output = user_repr

        else:
            long_short_term_user_hist = torch.cat([user_init, st_token_emb, embedded_hist, co_token_emb, order_repr], dim=1).contiguous().permute(1, 0, 2)
            
            
            src_mask = generate_square_subsequent_mask(long_short_term_user_hist.size(0))
            tgt_mask = generate_square_subsequent_mask(dq_token_emb.size(0))
            
            
            src_mask = src_mask.to(long_short_term_user_hist.device)
            tgt_mask = tgt_mask.to(long_short_term_user_hist.device)

            transformer_output = self.transformer(src=long_short_term_user_hist, 
                                    tgt=dq_token_emb, 
                                    src_mask=src_mask, 
                                    tgt_mask=tgt_mask)
            
            transformer_output = transformer_output.permute(1, 0, 2).squeeze()
            if len(transformer_output.size()) == 1:
                transformer_output = transformer_output.unsqueeze(0)
        
        # # process the regions' heat map (in Ablation)
        if not self.pre_train:
            dispatch_map = input_dict["obs"][1]
            dispatch_map_repr = self.map_encoder3(dispatch_map).squeeze()
            if len(dispatch_map_repr.size()) == 1:
                dispatch_map_repr = dispatch_map_repr.unsqueeze(0)


        # cat_feature = torch.cat([transformer_output, dispatch_map_repr], dim=-1)

        # dispatch map ablation
        # cat_feature = torch.cat([attention_repr, level_est], dim=-1)

        # self.inner_feature = self.fusion3(cat_feature)
        if not self.pre_train:
            cat_feature = torch.cat([transformer_output, dispatch_map_repr], dim=-1)
        else:
            cat_feature = transformer_output

        all_feature = self.fusion3(cat_feature)
        # all_feature = transformer_output

        if self.recurrent:
            all_feature = all_feature.unsqueeze(1)
            print(f"b1: {all_feature.size()}, {state[0].size()}")
            state = tuple([t.unsqueeze(0) if len(t.size())==2 else t for t in state])
            # assert all_feature.size(0) == state[0].size(1), f"expect same batch size, got {all_feature.size()} and {state[0].size()}"
            print(f"b2: {all_feature.size()}, {state[0].size()}")
            recurrent_output, next_state = self.state_encoder(all_feature, state)
            
            recurrent_output = recurrent_output.squeeze()
            if len(recurrent_output.size()) == 1:
                recurrent_output = recurrent_output.unsqueeze(0)

            next_state = [t.squeeze() for t in next_state]
            print(f"a: {recurrent_output.size()}, {next_state[0].size()}")
            self.inner_feature = recurrent_output
            
            return self.inner_feature, next_state
        else:
            
            self.inner_feature = all_feature
        # self.inner_feature = self.fusion(transformer_output)

            return self.inner_feature, state
    
    # def get_initial_state(self):
    #     if self.recurrent:
    #         return [torch.zeros([USER_EMBEDDING_SIZE]), torch.zeros([USER_EMBEDDING_SIZE])]
    #     else:
    #         return []
        
class RecurrentTDQN(LSTMWrapper, CDPModel_TDQN_with_pretrain):
    pass

RecurrentTDQN._wrapped_forward = CDPModel_TDQN_with_pretrain.forward

class DTWrappedRecurrentTDQN(DQNTorchModel, RecurrentTDQN):
    pass



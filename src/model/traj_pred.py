# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from src.model.losses import GaussianMixtureNLLLoss
from src.model.global_interactor import GlobalInteractor
from src.model.local_encoder import LocalEncoder
from src.model.decoder import MLPDecoder
from utils import TemporalData, rotate_back_prediction

class TrajPredEncoderDecoder(pl.LightningModule):
    """ Trajectory prediction model, adopted from HiVT.    
    """

    def __init__(self,
                 historical_steps: int = 25,
                 future_steps: int = 25,
                 num_modes: int = 5,
                 rotate: bool = True,
                 node_dim: int = 2,
                 add_agent_feats: bool = True,
                 edge_dim: int = 2,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 num_global_layers: int = 3,
                 local_radius: float = 50,
                 parallel: bool = False,
                 lr: float = 0.0001,
                 weight_decay: float = 0.001,
                 T_max: int = 64,
                 mask_target_agents: bool = True,
                 ) -> None:
        super(TrajPredEncoderDecoder, self).__init__()
        self.save_hyperparameters()
        self.rotate = rotate
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.mask_target_agents = mask_target_agents

        # network
        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          add_agent_feats=add_agent_feats,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)

        self.decoder = MLPDecoder(local_channels=embed_dim,
                                global_channels=embed_dim,
                                future_steps=future_steps,
                                num_modes=num_modes,
                                uncertain=True,
                                scale_dim=1)

        # loss
        self.reg_loss = GaussianMixtureNLLLoss(reduction='mean')

        # latent features
        self.h = None

        # predictions
        self.mu = None
        self.pi = None
        self.scale = None


    def forward(self, data: TemporalData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method performs one forward pass through the network.

        Return
            loc_scale: Mean and standard deviation of Gaussian mixture prediction [K, N, T, 3],
                with number of modes K, number of agents N and future time steps T.
            pi: Mixture coefficients [N, K], with number of modes K and number of agents N.
        """
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        # encoding
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)

        # latent features
        self.h = local_embed

        # decoding
        loc_scale, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return loc_scale, pi


    def training_step(self, data, batch_idx):
        """
        Method performs one forward pass and computes the regression loss.

        Args:
            data: current batch of data.   
            batch_index: index of current batch.     
        """

        # forward
        loc_scale, pi = self(data)
        y_hat, scale = loc_scale.chunk(2, dim=-1)

        # padding mask 
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
                
        # ground truth
        y = data.y

        # filter on valid target agents
        agent_index = data.agent_index[data.valid]

        # compute loss
        if self.mask_target_agents:
            assert len(agent_index) > 0, "At least one target agent needs to be valid."
            reg_loss = self.reg_loss(y_hat[:, agent_index], pi[agent_index], scale[:, agent_index], y[agent_index], reg_mask[agent_index])
        else:
            reg_loss = self.reg_loss(pi, y_hat, scale, y, reg_mask)

        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return reg_loss
    

    def validation_step(self, data, batch_idx):
        """
        Method predicts a distribution over future trajectories given the current batch.

        Args:
            data: current batch of data.
            batch_index: index of current batch.       
        """
        
        # forward
        loc_scale, pi = self(data)
        y_hat, scale = loc_scale.chunk(2, dim=-1)

        # log predictions
        self.mu = torch.swapaxes(rotate_back_prediction(y_hat, data.rotate_mat), 0, 1)  # N x K x T x 2
        self.pi = pi # N x K
        self.scale = torch.swapaxes(scale, 0, 1) # N x K x T x 1


    def get_predictions(self, agent_index: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method returns the current prediction given the target agent indices.
        Args:
            agent_index: List of agent indices of length [N].

        Return:
            y_hat_agent: Mean trajectories of all modes K [NxKxTx2]
            pi_agent: Mode coefficents [NxK]
            sigma_agent: Standard deviation of Gaussian mixture distributions [NxKxT]

        """

        mu = self.mu.detach().cpu().numpy()
        pi = self.pi.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()

        y_hat_agent = mu[agent_index, :, :, : 2]
        pi_agent = pi[agent_index]
        sigma_agent = scale[agent_index, :, :, 0] 

        return y_hat_agent, pi_agent, sigma_agent
    

    def get_latent_features(self) -> torch.Tensor:
        """
        Return the latent feature vetors.

        Return
            h: latent feature vectors [NxF], with the feature dimension F=128.

        """

        return self.h


    def configure_optimizers(self):
        """
        Configuration of the training optimizer.

        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

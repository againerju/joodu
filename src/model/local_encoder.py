# Code adopted from https://github.com/ZikangZhou/HiVT
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from src.model.embedding import MultipleInputEmbedding
from src.model.embedding import SingleInputEmbedding
from utils import DistanceDropEdge, TemporalData, init_weights

class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 add_agent_feats: bool,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False,
                ) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        self.add_agent_feats = add_agent_feats

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    add_agent_feats=add_agent_feats,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel,
                                    )
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout
                                    )

    def forward(self, data: TemporalData) -> torch.Tensor:
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
        if self.parallel:
            raise NotImplementedError()
        else:
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                if self.add_agent_feats:
                    out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                             bos_mask=data['bos_mask'][:, t],
                                             velocities=data.velocities[:, t], accelerations=data.accelerations[:, t],
                                             is_vehicle=data.is_vehicle[:], rotate_mat=data['rotate_mat'])
                else:
                    out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                            bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])

            out = torch.stack(out)  # [T, N, D]
                
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])


        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                            max_vels=data['max_vels'], give_ways=data['give_ways'],
                            availability=data['availability'], rotate_mat=data['rotate_mat'])

        return out


class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 add_agent_feats: bool,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.add_agent_feats = add_agent_feats
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        if self.add_agent_feats:
            # 4 Additional Agent features (2x velocity, 2x acceleration, 1x vehicle/pedestrian bool)
            node_dim = node_dim + 5

        # encode single agent
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)

        if self.add_agent_feats:
            self.nbr_embed = MultipleInputEmbedding(in_channels=[2, 2, 2, 1, edge_dim], out_channel=embed_dim)
        else:
            self.nbr_embed = MultipleInputEmbedding(in_channels=[2, edge_dim], out_channel=embed_dim)

        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)


    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                velocities: torch.Tensor = None,
                accelerations: torch.Tensor = None,
                is_vehicle: torch.Tensor = None,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None,
                ) -> torch.Tensor:
        
        if rotate_mat is None:
            if self.add_agent_feats:
                in_data = torch.cat([x, velocities, accelerations, is_vehicle.unsqueeze(-1)], dim=-1)
            else:
                in_data = x
            center_embed = self.center_embed(in_data)
        else:
            x_transformed = torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2)
            if self.add_agent_feats:
                in_data = torch.cat([
                    x_transformed,
                    torch.bmm(velocities.unsqueeze(-2), rotate_mat).squeeze(-2),
                    torch.bmm(accelerations.unsqueeze(-2), rotate_mat).squeeze(-2),
                    is_vehicle.unsqueeze(-1)
                ], dim=-1)
            else:
                in_data = x_transformed

            center_embed = self.center_embed(in_data)
        center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        
        if self.add_agent_feats:
            center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr,
                                                          rotate_mat, size, velocities=velocities, accelerations=accelerations,
                                                          is_vehicle=is_vehicle)
        else:
            center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr,
                                                          rotate_mat, size)

        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        
        return center_embed


    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                velocities_j=None,  # Appending a "_j" at the end is recognized by the MessagePassing class
                accelerations_j=None,
                is_vehicle_j=None) -> torch.Tensor:
        if rotate_mat is None:
            if self.add_agent_feats:
                nbr_embed = self.nbr_embed([x_j, velocities_j, accelerations_j, is_vehicle_j.unsqueeze(-1).float(), edge_attr])
            else:
                nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            
            if self.add_agent_feats:
                nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                            torch.bmm(velocities_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                            torch.bmm(accelerations_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                            is_vehicle_j.unsqueeze(-1).float(),
                                            torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
            else:
                nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                            torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        # linear layers for query q_i^t, key k_{ij}^t and value v_{ij}^t
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)  # query from center agent features, eq. (3)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)  # key from neighboring agent features, eq. (3)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)  # value from neighboring agent features, eq. (3)
        # scaled dot-product attention
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale  # unnormalized attention scores, eq. (4)
        alpha = softmax(alpha, index, ptr, size_i)  # normalized attention scores, eq. (4)
        alpha = self.attn_drop(alpha)  # dropout on attention scores = alpha_i^t
        return value * alpha.unsqueeze(-1)  # select values using attention scores, eq. (5) = m_i^t


    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))  # use a gate as soft-switch for message passing, eq. (6)
        return inputs + gate * (self.lin_self(center_embed) - inputs)  # = \hat{z}_i^t as weighted sum of center embeds z_i^t with self-weights 
                                                                       # and the message features m_i^t, eq. (7)


    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size,
                   velocities: torch.Tensor = None,
                   accelerations: torch.Tensor = None,
                   is_vehicle: torch.Tensor = None) -> torch.Tensor:
        
        if self.add_agent_feats:
            center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                        edge_attr=edge_attr, rotate_mat=rotate_mat, size=size,
                                                        velocities=velocities, accelerations=accelerations,
                                                        is_vehicle=is_vehicle))
        else:
            center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                        edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))  # learnable token
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))  # learnable position embedding
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)


    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1) 
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x


    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        
        self.max_vels_embed = nn.Linear(1, embed_dim)
        self.give_ways_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.availability_embed = nn.Parameter(torch.Tensor(5, embed_dim))
        nn.init.normal_(self.give_ways_embed, mean=0., std=.02)
        nn.init.normal_(self.availability_embed, mean=0., std=.02)
        
        self.apply(init_weights)


    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                max_vels: torch.Tensor = None,
                give_ways: torch.Tensor = None,
                availability: torch.Tensor = None,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x

        max_vels = max_vels.float()
        give_ways = give_ways.long()
        availability = availability.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, rotate_mat, size, 
                                            max_vels=max_vels, give_ways=give_ways, availability=availability)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor


    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                max_vels_j: torch.Tensor = None,
                give_ways_j: torch.Tensor = None,
                availability_j: torch.Tensor = None,
                rotate_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                [self.max_vels_embed(max_vels_j.unsqueeze(-1)),
                                self.give_ways_embed[give_ways_j],
                                self.availability_embed[availability_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                [self.max_vels_embed(max_vels_j.unsqueeze(-1)),
                                self.give_ways_embed[give_ways_j],
                                self.availability_embed[availability_j]])
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)


    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)


    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size,
                   max_vels: torch.Tensor = None,
                   give_ways: torch.Tensor = None,
                   availability: torch.Tensor = None) -> torch.Tensor:

        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                            max_vels=max_vels, give_ways=give_ways,
                                            availability=availability, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)


    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)

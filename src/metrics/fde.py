# Code adopted from https://github.com/ZikangZhou/HiVT
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class MinFDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MinFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                     process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        """Update metric.

        Args:
            pred: predicted trajectories [N, K, T, 2]
            target: ground truth trajectories [N, T, 2]
        """
        fde = compute_fde(pred, target)
        min_id = fde.argmin(dim=-1)
        self.sum += fde[torch.arange(len(min_id)), min_id].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


def compute_fde(pred, target):

    return torch.norm(pred[..., -1, :] - target[..., -1, :][:, None], p=2, dim=-1)

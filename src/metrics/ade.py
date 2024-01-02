# Code adopted from https://github.com/ZikangZhou/HiVT
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class MinADE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MinADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
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
        ade = compute_ade(pred, target)
        min_id = ade.argmin(dim=-1)
        self.sum += ade[torch.arange(len(min_id)), min_id].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


def compute_ade(pred, target):

    return torch.norm(pred - target[:, None], p=2, dim=-1).mean(dim=-1)

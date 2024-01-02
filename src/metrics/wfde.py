from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric
from src.metrics.fde import compute_fde


class WFDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(WFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                   process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               weights: torch.Tensor,
               target: torch.Tensor) -> None:
        """ Update metric.

        Args
            pred: predicted trajectories [N, K, T, 2]
            weights: sum up to one along K-dimension [N, K]
            target: ground truth trajectories [N, T, 2]
        """
        fde = compute_fde(pred, target)
        wfde = torch.mul(fde, weights)
        self.sum += wfde.sum(dim=-1).sum()
        self.count += weights.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

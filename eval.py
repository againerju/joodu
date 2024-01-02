import os
from argparse import ArgumentParser, Namespace
from typing import List, Optional
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from src.eval.evaluator import Evaluator
from src.data.shifts_datamodule import ShiftsDataModule
from src.data.shifts_dataset import ShiftsDataset
from src.model.e_reg import Ereg
from src.model.lgmm import LGMM
from src.model.traj_pred import TrajPredEncoderDecoder


def parse_command_line_arguments(args: Optional[List[str]] = None) -> Namespace:

    parser = ArgumentParser()
    parser.add_argument('--traj_pred_model_path', type=str, default="None")
    parser.add_argument('--ood_detection_model_path', type=str)
    parser.add_argument('--uncertainty_model_path', type=str)
    parser.add_argument('--split', type=str, default='eval')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser = ShiftsDataModule.add_shifts_specific_args(parser)
    args = parser.parse_args()

    return args


def main(args: Optional[List[str]] = None) -> None:

    pl.seed_everything(2022)

    # parse command line arguments
    args = parse_command_line_arguments(args)

    # experiment
    experiment_root = os.path.split(
        os.path.split(args.traj_pred_model_path)[0])[0]

    # trajectory prediction model
    traj_pred_model = TrajPredEncoderDecoder.load_from_checkpoint(
        checkpoint_path=args.traj_pred_model_path, parallel=False, **vars(args))
    traj_pred_model.eval()

    # latent Gaussian mixture model
    ood_model = LGMM(model_path=args.ood_detection_model_path)

    # error regression network
    uncertainty_model = Ereg.load_from_checkpoint(
        checkpoint_path=args.uncertainty_model_path, **vars(args))
    uncertainty_model.eval()

    # dataset
    val_dataset = ShiftsDataset(root=args.shifts_root, split=args.split,
                                local_radius=traj_pred_model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=True)

    # evaluator
    evaluator = Evaluator()
    evaluator.set_eval_path_from(experiment_root,
                                 kwargs={
                                     "split": args.split,
                                     "ood": "lgmm",
                                     "u": "e_reg"
                                 })

    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():

            # ground truth
            agent_index = batch.agent_index[batch.valid].detach(
            ).cpu().numpy().tolist()
            y = batch.y[agent_index].detach().cpu().numpy()
            alpha = batch.ood[agent_index].detach().cpu().numpy()

            # trajectory prediction
            traj_pred_model.validation_step(batch, i, eval=True)
            y_hat, pi, sigma = traj_pred_model.get_predictions(
                agent_index, numpy=True)
            y_hat, pi = evaluator.select_top_d_trajectories(y_hat, pi)

            # latent features
            h = traj_pred_model.get_latent_features()
            h = h[agent_index]

            # ood detection
            alpha_hat = ood_model.predict_ood_score(h.detach().cpu().numpy())

            # uncerainty estimation
            e_hat = uncertainty_model.predict_uncertainty(h)
            e_hat = e_hat.squeeze(1).cpu().numpy()

            # compute metrics for batch
            evaluator.compute_batch_metrics(
                y, y_hat, pi, sigma, alpha, alpha_hat, e_hat)

    # evaluate
    evaluator.full_eval()


if __name__ == "__main__":
    main()

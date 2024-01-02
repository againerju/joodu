import os
from pathlib import Path
import torch
from tqdm import tqdm

from src.data.shifts_datamodule import ShiftsDataModule
from ysdc_dataset_api.evaluation.metrics import compute_all_aggregator_metrics
from src.eval.evaluator import Evaluator


class JOODUDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, device="cuda"):
        self.root = root
        self.split = split
        self.device = device
        self.path = os.path.join(self.root, self.split + "_results")
        self.file_paths = [f for f in Path(
            self.path).rglob('*.pt') if f.is_file()]
        self.file_paths.sort()

    def get_uncertainty_target(self, data):
        """Return target for error regression network.
        """
        target = torch.tensor(data["metrics"]["weightedADE"])
        return torch.log(target + 1e-3)

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])

        item_data = dict()
        item_data["h"] = torch.tensor(data["h"])
        item_data["e"] = self.get_uncertainty_target(data)

        return item_data

    def __len__(self):
        return len(self.file_paths)


def joodu_collate_fn(batch):
    
    data = dict()
    for key in batch[0].keys():
        data[key] = torch.cat([item[key] for item in batch], dim=0)

    return data


def joodu_train_and_val_data(args, traj_pred_model, export_root):
    """Return train and validation data for JOODU.

    Return
        train_dataset: JOODUDataset
        val_dataset: JOODUDataset
    """
    # instantiate the datamodule with batch size 1
    args.train_batch_size = 1
    args.val_batch_size = 1
    args.num_workers = 0
    args.pin_memory = False
    args.persistent_workers = False
    args.shuffle = False
    datamodule = ShiftsDataModule.from_argparse_args(args)
    datamodule.setup()

    # create export paths
    export_root = os.path.split(
        os.path.split(args.traj_pred_model_path)[0])[0]
    export_paths = dict()
    for phase, phase_set in datamodule.get_phase_to_set_map(["train", "val"]).items():
        export_paths[phase] = os.path.join(
            export_root, f"{phase_set}_results")

    # if the prediction results are available, skip the prediction step
    # otherwise, generate the prediction results
    result_exists = dict()

    for phase, path in export_paths.items():
        num_files = 0
        if os.path.exists(path):
            num_files = len(
                [f for f in Path(path).rglob('*') if f.is_file()])
        # TODO: compare with expected number of files
        result_exists[phase] = num_files > 0

    # generate prediction results if result does not exist
    result_exists['train'] = False
    for phase, exists in result_exists.items():
        if not exists:
            print(
                f"Results for phase {phase} do not exist. Results are generated.")

            # initialize evaluator
            evaluator = Evaluator()

            # get dataloader
            dataloader = datamodule.get_dataloader_from_phase(phase)
            traj_pred_model.eval()

            # export path
            export_path = export_paths[phase]
            os.makedirs(export_path, exist_ok=True)

            # iterate through dataloader
            for i, batch in enumerate(tqdm(dataloader)):
                with torch.no_grad():

                    # ground truth
                    seq_id = batch.seq_id[0]
                    agent_index = batch.agent_index[batch.valid].detach(
                    ).cpu().numpy().tolist()
                    y = batch.y[agent_index].detach().cpu().numpy()
                    alpha = batch.ood[agent_index].detach().cpu().numpy()

                    # trajectory prediction
                    if agent_index:
                        traj_pred_model.validation_step(batch, eval=True)
                        y_hat, pi, sigma = traj_pred_model.get_predictions(
                            agent_index, numpy=True)
                        y_hat, pi = evaluator.select_top_d_trajectories(
                            y_hat, pi)

                        # compute trajectory prediction metrics
                        metrics = compute_all_aggregator_metrics(pi, y_hat, y)

                        # latent features
                        h = traj_pred_model.get_latent_features()
                        h = h[agent_index].detach().cpu().numpy()

                        # log results
                        results = dict()
                        results["y"] = y
                        results["y_hat"] = y_hat
                        results["pi"] = pi
                        results["sigma"] = sigma
                        results["h"] = h
                        results["alpha"] = alpha
                        results["metrics"] = metrics
                        results["seq_id"] = seq_id

                        # create file structure according to Shifts
                        export_path = os.path.join(
                            export_paths[phase], "{:03}".format(int(seq_id) // 1000))
                        os.makedirs(export_path, exist_ok=True)

                        # save results as .pt file
                        torch.save(results, os.path.join(
                            export_path, "{:06}.pt".format(int(seq_id))))

    # joodu dataloader
    train_dataset = JOODUDataset(
        root=export_root, split=datamodule.map_phase_to_set["train"])

    val_dataset = JOODUDataset(
        root=export_root, split=datamodule.map_phase_to_set["val"])

    return train_dataset, val_dataset

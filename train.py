from argparse import ArgumentParser
import os
import pprint
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.shifts_datamodule import ShiftsDataModule
from src.data.joodu_dataset import joodu_collate_fn, joodu_train_and_val_data
from src.model.traj_pred import TrajPredEncoderDecoder
from src.model.lgmm import LGMM
from src.model.e_reg import Ereg
from src.exp.experiment import setup_logger

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--traj_pred_model_path', type=str, default=None)
    parser.add_argument('--skip_ood_model_training', action="store_true")
    parser.add_argument('--skip_uncertainty_model_training',
                        action="store_true")
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--val_batch_size', type=int, default=12)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--monitor', type=str, default='val_wADE')
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--gpus', type=str, default='0',
                        help="e.g. '-1', '0', '4', '1, 2', '0, 1, 3' etc,\
                            see pytorch lightning definition: https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html")
    parser = TrajPredEncoderDecoder.add_model_specific_args(parser)
    parser = Ereg.add_model_specific_args(parser)
    parser = ShiftsDataModule.add_shifts_specific_args(parser)
    args = parser.parse_args()

    print("\n\n--- 1st Phase: Train Trajectory Prediction ---\n\n")

    # logger
    proj_root = os.getcwd()
    save_dir = os.path.join(proj_root, "experiments", "traj_pred")

    if args.traj_pred_model_path:
        # load trajectory prediction model from checkpoint path
        traj_pred_model_path = args.traj_pred_model_path
        traj_pred_model = TrajPredEncoderDecoder.load_from_checkpoint(
            checkpoint_path=traj_pred_model_path, parallel=False, **vars(args))

    else:
        # train trajectory prediction model

        # logging
        tb_logger, cmd_logger = setup_logger(proj_root)

        # print parameters
        pprint.pprint(vars(args))

        # enable multi-gpu training
        accelerator = 'gpu'
        if args.gpus not in ['0', '1']:
            gpus = args.gpus
        else:
            gpus = [int(args.gpus)]

        # model and trainer
        model_checkpoint = ModelCheckpoint(
            monitor=args.monitor, save_top_k=args.save_top_k, mode='min', filename="traj_pred-{epoch:02d}-{val_wADE:.2f}")
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[
                                                model_checkpoint], devices=gpus, accelerator=accelerator)
        traj_pred_model = TrajPredEncoderDecoder(**vars(args))

        # data
        datamodule = ShiftsDataModule.from_argparse_args(args)

        # train
        traj_pred_model.train()
        trainer.fit(traj_pred_model, datamodule)
        print("\nThe training of the trajectory prediction model has finished.")

        # load best model for phase 2
        args.traj_pred_model_path = trainer.checkpoint_callback.best_model_path
        traj_pred_model = TrajPredEncoderDecoder.load_from_checkpoint(
            checkpoint_path=args.traj_pred_model_path, parallel=False, **vars(args))

    # --- 2nd Phase ---
    # Train the latent Gaussian mixture model for out-of-distribution detection
    # and the error regression network for uncertainty estimation
    print("\n\n--- 2nd Phase: Train JOODU ---\n\n")

    if not args.skip_ood_model_training or not args.skip_uncertainty_model_training:

        # train and validation data
        joodu_data_dir = os.path.split(
            os.path.split(args.traj_pred_model_path)[0])[0]
        train_dataset, val_dataset = joodu_train_and_val_data(
            args, traj_pred_model, joodu_data_dir)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1024, shuffle=True, collate_fn=joodu_collate_fn, num_workers=1, pin_memory=False, persistent_workers=False)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1024, shuffle=False, collate_fn=joodu_collate_fn, num_workers=1, pin_memory=False, persistent_workers=False)

        # train ood model
        if not args.skip_ood_model_training:

            print("\n\n--- Train OOD Detection Model ---\n")

            # logging
            tb_logger, cmd_logger = setup_logger(
                proj_root, task="ood_detection", model_name="001_lgmm")

            # print experiment information
            print("\nTraining OOD detection model with the following parameters:")
            print(f"Training data: {train_dataset.path}")
            pprint.pprint(vars(args))

            # instantiate ood model
            ood_model = LGMM()

            # train ood model
            ood_model.train(train_dataloader)
            print("\nThe training of the OOD detection model has finished.")

            # save ood model
            ood_model.export_model(os.path.join(
                tb_logger.log_dir, "lgmm.joblib"))

            # stop logging
            cmd_logger.close()
            del cmd_logger
        else:
            print("\n Skip training OOD detection model.\n")

        # train uncertainty model
        if not args.skip_uncertainty_model_training:

            print("\n\n--- Train Uncertainty Estimation Model ---\n")

            # logging
            tb_logger, cmd_logger = setup_logger(
                proj_root, task="uncertainty_estimation", model_name="001_e_reg")

            # print experiment information
            print(
                "\nTraining uncertainty estimation model with the following parameters:")
            print(f"Training data: {train_dataset.path}")
            print(f"Validation data: {val_dataset.path}")
            pprint.pprint(vars(args))

            # instantiate uncertainty model
            model_checkpoint = ModelCheckpoint(
                monitor="val_uncertainty_loss", save_top_k=5, mode='min')
            uncertainty_model = Ereg()

            # train uncertainty model
            trainer = pl.Trainer(max_epochs=args.e_reg_max_epochs, logger=tb_logger, callbacks=[
                                 model_checkpoint], gpus=1)
            trainer.fit(uncertainty_model, train_dataloader, val_dataloader)
            print("\nThe training of the uncertainty estimation model has finished.")

            # stop logging
            cmd_logger.close()
        else:
            print("\n Skip training uncertainty estimation model.\n")

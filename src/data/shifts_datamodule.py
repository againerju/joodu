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
import os
from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from src.data.shifts_dataset import ShiftsDataset


class ShiftsDataModule(LightningDataModule):

    def __init__(self,
                 shifts_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 train_set_name: str = "train",
                 val_set_name: str = "dev",
                 test_set_name: str = "eval",
                 azure: bool = False) -> None:
        super(ShiftsDataModule, self).__init__()
        self.root = shifts_root
        self.azure = azure
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius
        self.map_phase_to_set = dict(
            train=train_set_name, val=val_set_name, test=test_set_name)

    def prepare_data(self) -> None:
        ShiftsDataset(
            self.root, self.map_phase_to_set["train"], self.train_transform, self.local_radius, self.azure)
        ShiftsDataset(
            self.root, self.map_phase_to_set["val"], self.val_transform, self.local_radius, self.azure)
        ShiftsDataset(
            self.root, self.map_phase_to_set["test"], self.val_transform, self.local_radius, self.azure)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ShiftsDataset(
            self.root, self.map_phase_to_set["train"], self.train_transform, self.local_radius, self.azure)
        self.val_dataset = ShiftsDataset(
            self.root, self.map_phase_to_set["val"], self.val_transform, self.local_radius, self.azure)
        self.test_dataset = ShiftsDataset(
            self.root, self.map_phase_to_set["test"], self.val_transform, self.local_radius, self.azure)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def get_dataloader_from_phase(self, phase: str):
        if phase == "train":
            return self.train_dataloader()
        elif phase == "val":
            return self.val_dataloader()
        elif phase == "test":
            return self.test_dataloader()
        else:
            raise ValueError("Invalid phase name: {}".format(phase))

    def get_phase_to_set_map(self, phases=["train"]):
        return {phase: self.map_phase_to_set[phase] for phase in phases}

    @staticmethod
    def add_shifts_specific_args(parent_parser):
        proj_root = os.getcwd()
        parser = parent_parser.add_argument_group('Shifts')
        parser.add_argument('--train_set_name', type=str, default="train")
        parser.add_argument('--val_set_name', type=str, default="dev")
        parser.add_argument('--test_set_name', type=str, default="eval")
        args = parent_parser.parse_args()
        if args.azure:
            shifts_root = os.path.join(os.environ["INPUT_DATADIR"], "Shifts")
        else:
            shifts_root = os.path.join(proj_root, "data", "Shifts")
        parser.add_argument('--shifts_root', type=str, default=shifts_root)
        return parent_parser

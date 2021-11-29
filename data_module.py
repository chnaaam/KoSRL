import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import SrlDataset

class SrlDataModule(pl.LightningDataModule):
    def __init__(
            self,
            vocab_path,
            tokenizer_type,
            batch_size,
            cache_path,
            is_shuffle=True,
            data_path=None,
            max_len=100,
            is_test=False,
            split_ratio={"train": 0.8, "valid": 0.1, "test": 0.1}):

        super().__init__()

        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.data_path = data_path
        self.pad_token = "[PAD]"
        self.max_len = max_len
        self.split_ratio = split_ratio
        self.tokens = []
        self.labels = []

        self.dataset = SrlDataset(
            vocab_path=self.vocab_path,
            tokenizer_type=tokenizer_type,
            data_path=self.data_path,
            cache_path=cache_path,
            max_len=self.max_len,
            is_test=is_test)

        self.len_train_set = int(len(self.dataset) * self.split_ratio["train"])
        self.len_test_set = int(len(self.dataset) * self.split_ratio["test"])
        self.len_valid_set = len(self.dataset) - self.len_train_set - self.len_test_set

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset,
            [self.len_train_set + self.len_valid_set, self.len_test_set])

        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [self.len_train_set, self.len_valid_set])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.is_shuffle,
            num_workers=0)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0)

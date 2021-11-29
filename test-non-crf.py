import os
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import configure
from data_module import SrlDataModule
from net.BiaffineSrlwithoutCRF import SpanSrlNet

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    config = configure()
    data_path = config.data_path
    model_path = config.model_path
    cache_path = config.cache_path
    vocab_path = config.vocab_path

    model_name = config.model_name
    lstm_hidden_size = config.lstm_hidden_size
    lstm_dropout = config.lstm_dropout
    num_layers = config.num_layers

    max_length = config.max_length
    batch_size = config.batch_size
    precision = config.precision

    epochs = config.epochs
    lr = config.lr
    weight_decay = config.weight_decay

    srl_data_module = SrlDataModule(
        vocab_path=vocab_path,
        tokenizer_type="wordpiece",
        batch_size=batch_size,
        cache_path=cache_path,
        data_path=data_path,
        max_len=max_length)

    model = SpanSrlNet(
        bert_name=model_name,
        lstm_hidden_size=lstm_hidden_size,
        # lstm_dropout=lstm_dropout,
        num_layers=num_layers,
        label_size=srl_data_module.dataset.len_labels,
        pad_id=srl_data_module.dataset.pad_id,
        max_length=max_length,
        t2i=srl_data_module.dataset.t2i,
        i2t=srl_data_module.dataset.i2t,
        i2l=srl_data_module.dataset.i2l,
        l2i=srl_data_module.dataset.l2i,
        label_i2l=None,
        lr=lr,
        weight_decay=weight_decay)

    # path = "models/srl-non-crf-biaffine-crf-mean-(-3)-fc2-256-16-epoch=3-f1_score=75.0721.ckpt"
    # check_point = torch.load(path)
    # model.load_state_dict(check_point["state_dict"])
    # torch.save(model.state_dict(), "./test-non-crf.model")

    path = "./test-non-crf.model"
    model.load_state_dict(torch.load(path))

    # ############################################################################
    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus="0",
        accelerator="dp",
        precision=precision,
        gradient_clip_val=0.5
    )

    # trainer.fit(model, srl_data_module)
    trainer.test(model, srl_data_module.test_dataloader())
    # ############################################################################

if __name__ == "__main__":
    main()
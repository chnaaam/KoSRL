import os
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import configure
from data_module import SrlDataModule
from net.conditional_bert import ConditionalBert

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    config = configure()
    data_path = config.data_path
    model_path = config.model_path
    cache_path = config.cache_path
    vocab_path = config.vocab_path

    model_name = config.model_name
    tokenizer_type = "wordpiece"
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
        tokenizer_type=tokenizer_type,
        batch_size=batch_size,
        cache_path=cache_path,
        data_path=data_path,
        max_len=max_length)

    model = ConditionalBert(
        bert_name=model_name,
        vocab_size=srl_data_module.dataset.len_vocab,
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
        lr=lr,
        weight_decay=weight_decay)

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",
        dirpath=model_path,
        filename=f"srl-conditional-bert-ce-{max_length}-{precision}-" + "{epoch}-{acc:0.4f}",
        save_top_k=2,
        mode="max")

    # ############################################################################
    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus="1",
        accelerator="dp",
        precision=precision,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5
    )

    trainer.fit(model, srl_data_module)
    # ############################################################################

if __name__ == "__main__":
    main()

import torch
import pytorch_lightning as pl

from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

CANDIDATE_PREDICATE_TAG = ["VV", "XSV", "VCP", "VCN", "VA", "XSA"]

class SrlNet(pl.LightningModule):
    def __init__(
            self,
            vocab_size,
            morph_size,
            word_embed_size,
            lstm_hidden_size,
            num_layers,
            lstm_dropout,
            max_length,
            label_size,
            pad_id,
            t2i, i2t,
            i2l, l2i,
            word_embedding_vector,
            lr=5e-5,
            weight_decay=0.1,
            batch_first=True):

        super().__init__()

        self.vocab_size = vocab_size
        self.label_size = label_size
        self.pad_id = pad_id
        self.t2i = t2i
        self.i2t = i2t
        self.i2l = i2l
        self.l2i = l2i
        self.lr = lr
        self.weight_decay = weight_decay

        self.word_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_size).from_pretrained(torch.FloatTensor(word_embedding_vector), freeze=True)

        self.pos_embed = nn.Embedding(
            num_embeddings=morph_size,
            embedding_dim=64)

        self.lstm = nn.GRU(
            input_size=word_embed_size + 64 + 1,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=batch_first,
            num_layers=num_layers,
            # dropout=lstm_dropout
        )

        self.fc = nn.Linear(lstm_hidden_size * 2, label_size)
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_tags=label_size, batch_first=batch_first)

    def forward(self, tokens, predicate_indicator, morphs=None, labels=None, mask=None):

        x1 = self.word_embed(tokens)
        x2 = self.pos_embed(morphs)

        x = torch.cat((x1, x2, predicate_indicator.unsqueeze(-1)), dim=-1)

        x, _ = self.lstm(x)
        x = self.fc(x)
        emissions = self.dropout(x)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, labels, mask), self.crf.decode(emissions, mask)

            return (-1) * log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, mask)

            return sequence_of_tags

    def training_step(self, batch, batch_idx):
        self.train()

        opt = self.optimizers()

        tokens, morphs, predicate_tensor, labels = batch

        crf_mask = (tokens != self.pad_id).bool()

        loss, pred_tags = self(
            tokens=tokens,
            predicate_indicator=predicate_tensor,
            morphs=morphs,
            labels=labels,
            mask=crf_mask)

        opt.zero_grad()

        # opt.step()



        # y_true = []
        # y_pred = []
        #
        # for idx, label in enumerate(labels):
        #     true = []
        #     pred = []
        #     for jdx in range(len(label)):
        #         if label[jdx] == self.pad_id:
        #             break
        #
        #         if pred_tags[idx][jdx] == self.pad_id:
        #             pred_tags[idx][jdx] = self.l2i["O"]
        #
        #         true.append(self.i2l[label[jdx].item()])
        #         pred.append(self.i2l[pred_tags[idx][jdx]])
        #
        #     y_true.append(true)
        #     y_pred.append(pred)
        #
        # print(y_true[0][:30])
        # print(y_pred[0][:30])
        #
        # score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
        # print("f1_score : ", score * 100)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, morphs, predicate_tensor, labels = batch

        crf_mask = (tokens != self.pad_id).bool()

        pred_tags = self(
            tokens=tokens,
            predicate_indicator=predicate_tensor,
            morphs=morphs,
            labels=None,
            mask=crf_mask)

        y_true = []
        y_pred = []

        for idx, label in enumerate(labels):
            true = []
            pred = []
            for jdx in range(len(label)):
                if label[jdx] == self.pad_id:
                    break

                if pred_tags[idx][jdx] == self.pad_id:
                    pred_tags[idx][jdx] = self.l2i["O"]

                true.append(self.i2l[label[jdx].item()])
                pred.append(self.i2l[pred_tags[idx][jdx]])

            y_true.append(true)
            y_pred.append(pred)

        print()
        print(y_true[0][:30])
        print(y_pred[0][:30])
        print()

        score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
        print("f1_score : ", score * 100)
        self.log("f1_score", score * 100)

        # return {"f1_score": score}

    def test_step(self, batch, batch_idx):
        self.eval()

        tokens, morphs, labels = batch

        pred_tags = self(
            x=tokens,
            m=morphs,
            labels=None,
            mask=None)

        token_list = []
        y_true = []
        y_pred = []

        for idx, label in enumerate(labels):
            token = []
            true = []
            pred = []
            for jdx in range(len(label)):
                if label[jdx] == self.pad_id:
                    break

                if pred_tags[idx][jdx] == self.pad_id:
                    pred_tags[idx][jdx] = self.l2i["O"]

                token.append(self.i2t[tokens[idx][jdx].item()])
                true.append(self.i2l[label[jdx].item()])
                pred.append(self.i2l[pred_tags[idx][jdx]])

            token_list.append(token)
            y_true.append(true)
            y_pred.append(pred)

        # for idx in range(len(token_list)):
        #     for jdx in range(len(token_list[idx])):
        #         print(token_list[idx][jdx], end=" ")
        #     print()
        #
        #     for jdx in range(len(y_true[idx])):
        #         print(y_true[idx][jdx], end=" ")
        #     print()
        #
        #     for jdx in range(len(y_pred[idx])):
        #         print(y_pred[idx][jdx], end=" ")
        #     print()

        score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
        # print("f1_score : ", score * 100)
        # self.log("f1_score", score * 100)

        return {"f1_score": score}

    def test_epoch_end(self, outputs):
        avg_f1 = torch.tensor([x['f1_score'] for x in outputs]).mean()

        # true_labels, pred_labels = [], []
        #
        # for x in outputs:
        #     true_labels += x["true_label"]
        #     pred_labels += x["pred_label"]
        # true_labels = [x["true_label"] for x in outputs]
        # pred_labels = [x["pred_label"] for x in outputs]

        print(avg_f1)
        self.log("f1_score", avg_f1)

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

import math
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer

from torchcrf import CRF

from transformers import BertModel

# from pytorch_lightning.metrics import F1
from seqeval.metrics import f1_score
from seqeval.scheme import IOBES
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class SpanSrlNet(pl.LightningModule):
    def __init__(
            self,
            bert_name,
            lstm_hidden_size,
            num_layers,
            label_size,
            pad_id,
            t2i,
            i2t,
            i2l,
            l2i,
            label_i2l,
            max_length,

            lr=5e-5, weight_decay=0.1, is_test=False):

        super().__init__()
        self.label_size = label_size
        self.max_length = max_length
        self.pad_id = pad_id
        self.t2i = t2i
        self.i2t = i2t
        self.l2i = l2i
        self.i2l = i2l
        self.label_i2l = label_i2l
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_test = is_test

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name)

        self.lstm = nn.GRU(
            # input_size=word_embed_size + pos_embed_size + 1,
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers)

        self.fc1 = nn.Linear(lstm_hidden_size * 2, label_size)

        self.dropout = nn.Dropout(0.5)

        self.crf = CRF(num_tags=label_size, batch_first=True)

    def forward(self, tokens, token_type_ids, attention_mask, labels=None, mask=None):
        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]

        x, _ = self.lstm(x)
        y = self.fc1(x)

        emissions = self.dropout(y)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, labels, mask), self.crf.decode(emissions, mask)

            return (-1) * log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, mask)

            return sequence_of_tags


    def training_step(self, batch, batch_idx):
        self.train()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()
        crf_mask = (tokens != self.pad_id).bool()

        loss, pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            labels=labels,
            mask=crf_mask)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            labels=None,
            mask=None)

        labels = labels.tolist()

        y_true = []
        y_pred = []

        for idx, label in enumerate(labels):
            true = []
            pred = []
            for jdx in range(len(label)):
                if label[jdx] == self.pad_id:
                    break

                if pred_y[idx][jdx] == self.pad_id:
                    pred_y[idx][jdx] = self.l2i["O"]

                true.append(self.i2l[label[jdx]])
                pred.append(self.i2l[pred_y[idx][jdx]])

            y_true.append(true)
            y_pred.append(pred)

        score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
        print(score)
        self.log("f1_score", score * 100)

    def test_step(self, batch, batch_idx):

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            labels=None,
            mask=None)

        if self.is_test:

            tokens = tokens.tolist()
            labels = labels.tolist()

            y_pred = []

            for idx, token in enumerate(tokens):
                true = []
                pred = []
                t = []
                for jdx in range(len(token)):
                    if tokens[idx][jdx] == self.pad_id:
                        break

                    t.append(tokens[idx][jdx])
                    if pred_y[idx][jdx] == self.pad_id:
                        pred_y[idx][jdx] = self.l2i["O"]

                    pred.append(self.i2l[pred_y[idx][jdx]])

                decoded = self.decode(t, pred)

                y_pred.append(decoded)

            y_true = []
            for label in labels:
                true = []
                for l in label:
                    if l == self.pad_id:
                        break

                    true.append(self.label_i2l[l])
                y_true.append(true)


            score = f1_score(y_true, y_pred)
            print(score * 100)

            return {
                "f1_score": score * 100,
                "true_label": y_true,
                "pred_label": y_pred}

        else:
            labels = labels.tolist()

            y_true = []
            y_pred = []

            for idx, label in enumerate(labels):
                true = []
                pred = []
                for jdx in range(len(label)):
                    if label[jdx] == self.pad_id:
                        break

                    if pred_y[idx][jdx] == self.pad_id:
                        pred_y[idx][jdx] = self.l2i["O"]

                    true.append(self.i2l[label[jdx]])
                    pred.append(self.i2l[pred_y[idx][jdx]])

                y_true.append(true)
                y_pred.append(pred)

            score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
            print(score)
            self.log("f1_score", score * 100)

    def test_epoch_end(self, outputs):
        avg_acc = torch.tensor([x['f1_score'] for x in outputs]).mean()
        print(avg_acc)

        if self.is_test:
            true_labels = [x["true_label"] for x in outputs]
            pred_labels = [x["pred_label"] for x in outputs]

            array = [[] for _ in range(len(self.label_i2l.keys()))]
            for i in range(len(array)):
                array[i] = [0 for _ in range(len(self.label_i2l.keys()))]

            l2i = {l: i for i, l in self.label_i2l.items()}
            l2i = dict(sorted(l2i.items(), key=lambda x: x[0]))
            # del l2i["[PAD]"]

            ll = {k: i for i, k in enumerate(l2i.keys())}

            for trues, preds in zip(true_labels, pred_labels):
                for ts, ps in zip(trues, preds):
                    for t, p in zip(ts, ps):
                        t = ll[t]
                        p = ll[p]

                        array[p][t] += 1

            for i in range(len(array)):
                del array[i][-1]

            del array[-1]
            del ll['[PAD]']
            df_cm = pd.DataFrame(array, index=sorted([i for i in ll.keys()]),
                                 columns=sorted([i for i in ll.keys()]))
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, fmt="d", robust=True, vmin=0, vmax=100, annot_kws={"size": 10})
            # plt.show()
            plt.savefig("./output.png")
            self.log("f1_score", avg_acc)

    def decode(self, token_list, label_list):
        # sentence = "피고인은 거제에서 400만 원 상당의 순금목걸이를 피해자로부터 강취하였다."
        # token_list = ['피', '##고', '##인은', '거', '##제', '##에서', '400', '##만', '원', '상', '##당', '##의', '순', '##금', '##목', '##걸', '##이를', '피', '##해', '##자로', '##부터', '강', '##취', '##하였다', '.']
        # label_list = ['B-ARG0', 'I-ARG0', 'I-ARG0', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'O', 'O', 'O', 'O']

        token_list = self.tokenizer.convert_ids_to_tokens(token_list)
        word_list = self.tokenizer.convert_tokens_to_string(token_list).split(" ")
        result = []
        for word in word_list:
            l = "O"

            while word:
                if not label_list:
                    break

                token = token_list[0]
                label = label_list[0]

                if label != "O":
                    l = label[2:]

                if "##" in token:
                    token = token[2:]

                if word == token:
                    word = ""
                    del token_list[0]
                    del label_list[0]

                elif token in word:
                    word = word[len(token):]
                    del token_list[0]
                    del label_list[0]

            result.append(l)

        return result

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}]

        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay)

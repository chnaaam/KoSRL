import math
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchcrf import CRF

from transformers import BertModel

# from pytorch_lightning.metrics import F1
from seqeval.metrics import f1_score
# from seqeval.scheme import IOBES
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix

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
            max_length,

            lr=5e-5, weight_decay=0.1):

        super().__init__()
        self.label_size = label_size
        self.max_length = max_length
        self.pad_id = pad_id
        self.t2i = t2i
        self.i2t = i2t
        self.l2i = l2i
        self.i2l = i2l
        self.lr = lr
        self.weight_decay = weight_decay

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_name)

        self.start_fc = nn.Linear(self.bert.config.hidden_size, label_size)
        self.end_fc = nn.Linear(self.bert.config.hidden_size, label_size)

        self.biaffine_layer = BiAffine(label_size, label_size)
        self.squeeze_fc = nn.Linear(self.max_length, 1)

        self.fc = nn.Linear(label_size, label_size)

        self.crf = CRF(num_tags=label_size, batch_first=True)

    def forward(self, tokens, token_type_ids, attention_mask, labels=None, mask=None):
        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]

        start_x = self.start_fc(x)
        end_x = self.end_fc(x)

        y = self.biaffine_layer(start_x, end_x)

        y = y.transpose(-1, -2)
        y = F.relu(y)
        y = self.squeeze_fc(y)
        y = y.squeeze(-1)

        y = self.fc(y)

        emissions = y  # self.dropout(x)

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

        # loss = F.cross_entropy(
        #     pred_y.view(-1, self.label_size),
        #     labels.view(-1))

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

        # pred_y = torch.argmax(pred_y, dim=-1).tolist()
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

        score = f1_score(y_true, y_pred)
        print(score)
        self.log("f1_score", score * 100)

        # pred_y = F.softmax(pred_y, dim=-1)
        #
        # score = self.f1(torch.argmax(pred_y, dim=-1), labels)
        # self.log("f1_score", score.item() * 100)

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


class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_normal_(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1).transpose(1, 3)
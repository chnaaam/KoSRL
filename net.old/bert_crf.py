import math
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchcrf import CRF

from transformers import BertModel
from transformers import BertTokenizer

from seqeval.metrics import f1_score
from seqeval.scheme import IOBES
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class SrlNet(pl.LightningModule):
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
            label_i2l,
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
        self.label_i2l = label_i2l

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name)

        self.fc = nn.Linear(self.bert.config.hidden_size, label_size)

        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_tags=label_size, batch_first=True)

        # self.dice_loss = SelfAdjDiceLoss()
        # self.focal_loss = FocalLoss()
    def forward(self, tokens, token_type_ids, attention_mask, labels=None, crf_mask=None):

        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        x = self.fc(x)

        emissions = self.dropout(x)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, labels, crf_mask), self.crf.decode(emissions)

            return (-1) * log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, crf_mask)

            return sequence_of_tags

        # return x
        # # Encoder Part
        # encoded_x, hidden_state = self.encoder(tokens, token_type_ids, attention_mask)
        #
        # # Decoder Part
        # decoded_x = self.decoder(tokens, hidden_state)
        #
        # return decoded_x

    def training_step(self, batch, batch_idx):
        self.train()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()
        crf_mask = (tokens != self.pad_id).bool()

        loss, pred_tags = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            labels=labels,
            crf_mask=crf_mask)

        # loss = F.cross_entropy(pred_tags.view(-1, self.label_size), labels.view(-1))
        # loss = self.focal_loss(pred_tags.view(-1, self.label_size), labels.view(-1))

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_tags = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask)

        # pred_tags = torch.argmax(pred_tags, dim=-1).tolist()

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


        score = f1_score(y_true, y_pred)
        print(score)
        self.log("f1_score", score * 100)

        # return {"f1_score": score}

    def test_step(self, batch, batch_idx):

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            labels=None,
            crf_mask=None)

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

    def test_epoch_end(self, outputs):
        avg_acc = torch.tensor([x['f1_score'] for x in outputs]).mean()

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
        plt.show()
        self.log("f1_score", avg_acc)

    def decode(self, token_list, label_list):
        # sentence = "???????????? ???????????? 400??? ??? ????????? ?????????????????? ?????????????????? ???????????????."
        # token_list = ['???', '##???', '##??????', '???', '##???', '##??????', '400', '##???', '???', '???', '##???', '##???', '???', '##???', '##???', '##???', '##??????', '???', '##???', '##??????', '##??????', '???', '##???', '##?????????', '.']
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
             'weight_decay_rate': 0.01},
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

class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=1, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


class LabelLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LabelLayer, self).__init__()

        self.fc = nn.Linear(in_features=dim_in, out_features=dim_out)

    def forward(self, X):
        X = self.fx(X)

        return X
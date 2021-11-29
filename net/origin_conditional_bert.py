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

class ConditionalBert(pl.LightningModule):
    def __init__(
            self,
            bert_name,
            vocab_size,
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
        self.vocab_size = vocab_size
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

        # self.rnn = nn.GRU(
        #     input_size=self.bert.config.hidden_size,
        #     hidden_size=lstm_hidden_size,
        #     num_layers=num_layers,
        #     # dropout=lstm_dropout,
        #     bidirectional=True,
        #     batch_first=True)


        # self.start_logit = nn.Linear(self.bert.config.hidden_size, self.max_length)
        # self.end_logit = nn.Linear(self.bert.config.hidden_size, self.max_length)

        self.start_logit = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_logit = nn.Linear(self.bert.config.hidden_size, 1)

        # self.start_logit = nn.Linear(lstm_hidden_size * 2, 1)
        # self.end_logit = nn.Linear(lstm_hidden_size * 2, 1)

        self.dropout = nn.Dropout(0.5)

        # self.loss = FocalLoss()

    def forward(self, tokens, token_type_ids, attention_mask):
        device_type = tokens.device

        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]

        # x, _ = self.rnn(x)

        l1 = self.start_logit(x)
        l2 = self.end_logit(x)

        l1 = l1.squeeze(-1)
        l2 = l2.squeeze(-1)
        # l1 = self.start_logit(x[:,0,:])
        # l2 = self.end_logit(x[:,0,:])

        # l1 = l1.transpose(0, 1)
        # l2 = l2.transpose(0, 1)

        # l1 = self.dropout(l1)
        # l2 = self.dropout(l2)

        return l1, l2

    def training_step(self, batch, batch_idx):
        self.train()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        start_logit, end_logit = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask)

        # Cross Entropy
        loss1 = F.cross_entropy(start_logit, labels[:, 0])
        loss2 = F.cross_entropy(end_logit, labels[:, 1])

        # Focal Loss
        # loss1 = self.loss(start_logit, labels[:, 0])
        # loss2 = self.loss(end_logit, labels[:, 1])

        # L1 Loss
        # start_logit = F.log_softmax(start_logit, 1)
        # end_logit = F.log_softmax(end_logit, 1)
        #
        # start_logit = torch.argmax(start_logit, dim=-1)
        # end_logit = torch.argmax(end_logit, dim=-1)
        # start_labels = torch.nn.functional.one_hot(labels[:, 0], num_classes=self.max_length)
        # end_labels = torch.nn.functional.one_hot(labels[:, 1], num_classes=self.max_length)

        # loss_interval = end_logit - start_logit

        # For debugging
        # start_logit = torch.argmax(start_logit, dim=-1).tolist()
        # end_logit = torch.argmax(end_logit, dim=-1).tolist()
        # labels = labels.tolist()
        # print("labels : ", labels[0], " start/end : ", start_logit[0], end_logit[0])

        return loss1 + loss2 #* loss_interval


    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        start_logit, end_logit = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask)

        start_logit = F.softmax(start_logit, dim=-1)
        end_logit = F.softmax(end_logit, dim=-1)

        start_logit = torch.argmax(start_logit, dim=-1).tolist()
        end_logit = torch.argmax(end_logit, dim=-1).tolist()
        labels = labels.tolist()

        total_count = 0
        acc_count = 0

        for idx, (sl, el) in enumerate(zip(start_logit, end_logit)):
            if labels[idx][0] == sl and labels[idx][1] == el:
                acc_count += 1
            total_count += 1

        # loss1 = F.cross_entropy(start_logit, labels[:, 0])
        # loss2 = F.cross_entropy(end_logit, labels[:, 1])
        # score = (loss1 + loss2).item()

        acc = acc_count / total_count * 100
        print("Acc : ", acc)
        self.log("acc", acc)

        # pred_y = F.softmax(pred_y, dim=-1)
        #
        # score = self.f1(torch.argmax(pred_y, dim=-1), labels)
        # self.log("f1_score", score.item() * 100)

    def test_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, space_idx_list, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_tags = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            space_idx_list=space_idx_list)

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

        score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)

        return {
            "f1_score": score * 100,
            "y_true": y_true,
            "y_pred": y_pred}


    def test_epoch_end(self, outputs):
        avg_f1 = torch.tensor([x['f1_score'] for x in outputs]).mean()
        print(avg_f1)
        y_trues = [x["y_true"] for x in outputs]
        y_preds = [x["y_pred"] for x in outputs]

        label_list = list(self.l2i.keys())

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sn
        # import numpy as np
        # cm = np.zeros(shape=(len(label_list), len(label_list)))
        y_t, y_p = [], []
        for y_true, y_pred in zip(y_trues, y_preds):
            for t, p in zip(y_true, y_pred):

                y_t += t
                y_p += p
                # cm += confusion_matrix(t, p, labels=label_list)


        cm = confusion_matrix(y_t, y_p, labels=label_list)
        df_cm = pd.DataFrame(cm, index=[i for i in label_list],
                             columns=[i for i in label_list])
        plt.figure(figsize=(24, 20))


        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
        sn.set(font_scale=0.5)
        ax = sn.heatmap(df_cm, vmax=500, cmap="YlGnBu")

        for i in range(len(label_list)+1):
            ax.axvline(i, color="black", lw=2)
            ax.axhline(i, color="black", lw=2)
        plt.show()

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


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss

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
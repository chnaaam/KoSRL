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

        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.feature_fc = nn.Linear(self.bert.config.hidden_size, 64)
        self.span_fc = nn.Linear(64, label_size)

        # self.span_fc2 = nn.Linear(512, label_size, bias=False)
        # self.focal_loss = FocalLoss()
        # self.dice_loss = SelfAdjDiceLoss()

    def forward(self, tokens, token_type_ids, attention_mask, predicate_word_idx, space_idx_list:list, labels=None, crf_mask=None):
        device_type = tokens.device

        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        # x = self.feature_fc(x)

        previous_space_idx = 0

        return y


    def training_step(self, batch, batch_idx):
        self.train()

        tokens, predicate_labels, predicate_word_idx, space_idx_list, labels = batch

        attention_mask = (tokens != self.pad_id).float()
        crf_mask = (tokens != self.pad_id).bool()

        pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            predicate_word_idx=predicate_word_idx,
            space_idx_list=space_idx_list.tolist(),
            labels=labels,
            crf_mask=crf_mask)

        loss = F.cross_entropy(pred_y.view(-1, self.label_size), labels.view(-1))

        # loss = self.focal_loss(pred_y.view(-1, self.label_size), labels.view(-1))
        # loss = self.dice_loss(pred_y.view(-1, self.label_size), labels.view(-1))

        # pred_y = pred_y.view(-1, self.label_size)
        # loss = F.nll_loss(pred_y, labels)

        pred_y = torch.log_softmax(pred_y, dim=-1)
        pred_y = torch.argmax(pred_y, dim=-1).tolist()
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

        # print(y_true[0][:30])
        # print(y_pred[0][:30])

        score = f1_score(y_true, y_pred, mode="strict")
        print(score)

        # loss = self.focal_loss(pred_tags.view(-1, self.label_size), labels.view(-1))

        # pred_y = F.softmax(pred_y, dim=-1)
        #
        # score = self.f1(torch.argmax(pred_y, dim=-1), labels)
        # print("F1: ", score.item() * 100)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, predicate_word_idx, space_idx_list, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        pred_y = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask,
            predicate_word_idx=predicate_word_idx,
            space_idx_list=space_idx_list.tolist())

        pred_y = torch.argmax(pred_y, dim=-1).tolist()
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


class LabelLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LabelLayer, self).__init__()

        self.fc = nn.Linear(in_features=dim_in, out_features=dim_out)

    def forward(self, X):
        X = self.fx(X)

        return X

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
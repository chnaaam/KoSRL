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

        self.start_logit = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_logit = nn.Linear(self.bert.config.hidden_size + 1, 1)

        self.confidence_logit = nn.Linear(self.bert.config.hidden_size, 2)

        self.dropout = nn.Dropout(0.5)
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.loss = FocalLoss()

    def forward(self, tokens, token_type_ids, attention_mask):
        x = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = x[0]
        cls_state = last_hidden_state[:, 0, :]

        score = self.confidence_logit(cls_state)

        l1 = self.start_logit(last_hidden_state)
        logits = torch.cat([l1, last_hidden_state], dim=-1)
        l2 = self.end_logit(logits)

        l1 = l1.squeeze(-1)
        l2 = l2.squeeze(-1)

        score = self.dropout(score)
        l1 = self.dropout(l1)
        l2 = self.dropout(l2)

        return score, l1, l2

    def training_step(self, batch, batch_idx):
        self.train()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        confidence_score_logit, start_logit, end_logit = self(
            tokens=tokens,
            token_type_ids=predicate_labels,
            attention_mask=attention_mask)

        cs = torch.zeros(size=confidence_score_logit.shape, device=tokens.device)
        cs[labels != 0] = 1

        bc_loss = self.bce_loss(confidence_score_logit, cs)

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

        # For debugging
        # start_logit = torch.argmax(start_logit, dim=-1).tolist()
        # end_logit = torch.argmax(end_logit, dim=-1).tolist()
        # confidence_score_logit = F.softmax(confidence_score_logit, dim=-1)
        # confidence_score_idx = torch.argmax(confidence_score_logit, dim=-1).tolist()

        # labels = labels.tolist()
        # print("labels : ", labels[0], " start/end : ", start_logit[0], end_logit[0], ' class : ', confidence_score_idx[0])
        # print("labels : ", labels[0], " start/end : ", start_logit[0], end_logit[0])

        return 0.5 * (loss1 + loss2) + 0.5 * bc_loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        tokens, predicate_labels, labels = batch

        attention_mask = (tokens != self.pad_id).float()

        confidence_score_logit, start_logit, end_logit = self(
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

        acc = acc_count / total_count * 100
        print("Acc : ", acc)
        self.log("acc", acc)

    def test_step(self, batch, batch_idx):
        self.eval()

        pass


    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.weight_decay},
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
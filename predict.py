import os
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import configure
from data_module import SrlDataModule
from net.BiaffineSrl import SpanSrlNet

from eunjeon import Mecab

mecab = Mecab()

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
        lr=lr,
        weight_decay=weight_decay)

    # path = "models/srl-conditional-bert-ce-256-16-epoch=5-acc=82.1682.ckpt"
    # check_point = torch.load(path)
    # model.load_state_dict(check_point["state_dict"])
    #
    # torch.save(model.state_dict(), "./test.model")

    path = "./test.model"
    model.load_state_dict(torch.load(path))

    # [
    #   [
    #       ('카터는', 'ARG0'), ('역삼에서', 'ARG2'), ('카카오브레인으로', 'ARG3'), ('출근했었다.', 'PREDICATE')
    #   ],
    #   [
    #       ('천안에', 'ARG2'), ('카터는', 'ARG0'), ('살던', 'PREDICATE')
    #   ]
    # ]
    sentence = "하지만, 천안에 살던 카터는 역삼역에서 카카오브레인으로 출근했었다."
    # sentence ="1330년대 초 중국에서도 흑사병이 돌기 시작했으며 1334년 허베이에서 흑사병이 번졌으며, 유럽에서도 흑사병의 위세가 한창이던 1348년 – 1354년 동안 중국 각지에서도 흑사병 확산이 있었다."
    # sentence = "이런 메사들의 집합을 어떤 애니메이션 방식과 결합시키는 것이 캐릭터의 표현 방식과 한계가 정해져 있다."
    # sentence = "캐릭터를 표현하는 방법은 결국 애니메이션 기법과 밀접하게 연관되어 있고, 독자들이나 필자가 디자이너라면 캐릭터를 미술이나 예술적인 관점으로 보았다."

    token_list = srl_data_module.dataset.tokenizer.tokenize(sentence, type="wordpiece")
    predicate_tag_list = ["VV", "XSV", "VCP", "VCN", "VA", "XSA"]
    predicate_list = []
    tagged_pos = mecab.pos(sentence)
    words = sentence.split(" ")

    for idx, word in enumerate(words):
        while word:
            token, tag = tagged_pos[0]

            is_predicate = False

            for predicate in predicate_tag_list:
                if predicate in tag:
                    is_predicate = True
                    break

            if is_predicate:
                predicate_list.append({"text": words[idx], "idx": idx})

            word = word[len(token):]
            del tagged_pos[0]

    # predicate_list = list(set(predicate_list))

    for predicate in predicate_list:
        predicate_idx = predicate["idx"]
        predicate_start_idx, predicate_end_idx = 0, 0

        for word_idx, word in enumerate(sentence.split(" ")):
            tokens = srl_data_module.dataset.tokenizer.tokenize(word, type="wordpiece")

            if word_idx == predicate_idx:
                predicate_end_idx = predicate_start_idx + len(tokens) - 1
                break

            predicate_end_idx = predicate_start_idx + len(tokens) - 1
            predicate_start_idx += len(tokens)


        tokens = []
        for t in token_list:
            if t in srl_data_module.dataset.t2i:
                tokens.append(srl_data_module.dataset.t2i[t])
            else:
                tokens.append(srl_data_module.dataset.t2i[srl_data_module.dataset.unknown_token])

        predicate_tensor = [0 for _ in range(len(tokens))]
        for i in range(predicate_start_idx, predicate_end_idx + 1):
            predicate_tensor[i] = 1

        tokens = torch.tensor(tokens)
        predicate_tensor = torch.tensor(predicate_tensor)

        tokens = torch.unsqueeze(tokens, 0)
        predicate_tensor = torch.unsqueeze(predicate_tensor, 0)
        attention_mask = (tokens != srl_data_module.dataset.pad_id).float()

        with torch.no_grad():
            tokens = tokens#.to("cuda:0")
            predicate_tensor = predicate_tensor#.to("cuda:0")
            attention_mask = attention_mask#.to("cuda:0")

            model#.to("cuda:0")
            model.eval()

            pred_y = model(tokens=tokens, token_type_ids=predicate_tensor, attention_mask=attention_mask)
            print(pred_y)

            y_pred = []
            for idx, label in enumerate(pred_y[0]):
                y_pred.append(srl_data_module.dataset.i2l[label])

        print(y_pred)
        print()
        print()




if __name__ == "__main__":
    main()

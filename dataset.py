import os
import json
import pickle
import torch
from tqdm import tqdm

from torch.utils.data import Dataset

from ai_hub_dataset import AiHubDataset
from tokenizer import Tokenizer

class SrlDataset(Dataset):
    def __init__(
            self,
            vocab_path,
            tokenizer_type,
            cache_path=None,
            data_path=None,
            max_len=256,
            is_test=False):

        self.vocab_path = vocab_path
        self.cache_path = cache_path
        self.data_path = data_path
        self.max_len = max_len
        self.is_test = is_test
        self.pad_token = "[PAD]"
        self.unknown_token = "[UNK]"
        self.tokenizer = Tokenizer()

        if self.data_path and self.cache_path:
            if os.path.isfile(os.path.join(self.cache_path, "cached.data")):
                with open(os.path.join(self.cache_path, "cached.data"), "rb") as fp:
                    cached_data = pickle.load(fp)

                    self.token_list = cached_data["token_list"]
                    self.predicate_idx = cached_data["predicate_idx"]
                    self.arg_label_list = cached_data["arg_label_list"]

            else:
                self.corpus = AiHubDataset(data_path=data_path)
                self.token_list = []
                self.predicate_idx = []
                self.arg_label_list = []

                for data in tqdm(self.corpus.dataset):
                    sentence = data["text"]
                    va_pair = data["verb_arg_pair"]
                    predicate_info = va_pair["verb"]
                    args_info = va_pair["args"]

                    token_list, predicate_idx, arg_label_list = self.tokenizer(sentence, predicate_info, args_info)

                    self.token_list.append(token_list)
                    self.predicate_idx.append(predicate_idx)
                    self.arg_label_list.append(arg_label_list)

                with open(os.path.join(self.cache_path, "cached.data"), "wb") as fp:
                    pickle.dump({
                        "token_list": self.token_list,
                        "predicate_idx": self.predicate_idx,
                        "arg_label_list": self.arg_label_list,
                    }, fp)

        self.t2i = dict(self.tokenizer.tokenizer_wordpiece.vocab)
        self.i2t = {v: k for k, v in self.t2i.items()}
        self.len_vocab = len(self.tokenizer.tokenizer_wordpiece.vocab)


        if os.path.isfile(os.path.join(self.vocab_path, "vocab.json")):
            with open(os.path.join(self.vocab_path, "vocab.json"), 'r') as fp:
                data = json.load(fp)

                self.l2i = data["label"]
                self.i2l = {v: k for k, v in self.l2i.items()}
                self.label_vocab = self.l2i.keys()

        else:
            arg_label_vocab = ["O"]

            for arg_label_list in self.arg_label_list:
                for label in arg_label_list:
                    type = label["type"]

                    arg_label_vocab.append("B-" + type)
                    arg_label_vocab.append("I-" + type)
                    arg_label_vocab.append("E-" + type)
                    arg_label_vocab.append("S-" + type)

            self.arg_label_vocab = list(set(arg_label_vocab))

            self.l2i = {t: i + 1 for i, t in enumerate(self.arg_label_vocab)}
            self.l2i.setdefault(self.pad_token, 0)
            self.i2l = {v: k for k, v in self.l2i.items()}

            with open(os.path.join(self.vocab_path, "vocab.json"), 'w') as fp:
                json.dump({"label": self.l2i}, fp)

        self.len_labels = len(self.l2i)
        self.pad_id = self.l2i[self.pad_token]

        if self.is_test:

            if os.path.isfile(os.path.join(self.vocab_path, "vocab-arg.json")):
                with open(os.path.join(self.vocab_path, "vocab-arg.json"), 'r') as fp:
                    data = json.load(fp)

                    self.label_l2i = data["label"]
                    self.label_i2l = {v: k for k, v in self.label_l2i.items()}
                    self.label_label_vocab = self.label_l2i.keys()

            else:
                arg_label_vocab = ["O"]

                for arg_label_list in self.arg_label_list:
                    for label in arg_label_list:
                        type = label["type"]

                        arg_label_vocab.append(type)

                self.arg_label_vocab = list(set(arg_label_vocab))

                self.label_l2i = {t: i + 1 for i, t in enumerate(self.arg_label_vocab)}
                self.label_l2i.setdefault(self.pad_token, 0)
                self.label_i2l = {v: k for k, v in self.label_l2i.items()}

                with open(os.path.join(self.vocab_path, "vocab-arg.json"), 'w') as fp:
                    json.dump({"label": self.label_l2i}, fp)

            # self.len_labels = len(self.l2i)
            # self.pad_id = self.l2i[self.pad_token]

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        if self.is_test == False:
            return self.__train(idx)

        else:
            return self.__test(idx)

    def __train(self, idx):
        token_list = self.token_list[idx]
        predicate_start_idx, predicate_end_idx = self.predicate_idx[idx]
        arg_label_list = self.arg_label_list[idx]

        # ##################################################################################################
        tokens = []
        for t in token_list:
            if t in self.t2i:
                tokens.append(self.t2i[t])
            else:
                tokens.append(self.t2i[self.unknown_token])

        predicate_tensor = [0 for _ in range(self.max_len)]
        for i in range(predicate_start_idx, predicate_end_idx + 1):
            predicate_tensor[i] = 1

        arg_labels = [self.l2i["O"]] * self.max_len
        for label in arg_label_list:
            s_idx = label["start_idx"]
            e_idx = label["end_idx"]
            type = label["type"]

            if s_idx >= self.max_len:
                continue

            if s_idx == e_idx:
                arg_labels[s_idx] = self.l2i["S-" + type]

            else:
                for i in range(s_idx, e_idx + 1):
                    if i == s_idx:
                        arg_labels[i] = self.l2i["B-" + type]
                    elif i == e_idx:
                        arg_labels[i] = self.l2i["E-" + type]
                    else:
                        arg_labels[i] = self.l2i["I-" + type]

        for i in range(len(tokens), self.max_len):
            arg_labels[i] = self.pad_id

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        elif len(tokens) < self.max_len:
            tokens = tokens + [self.t2i[self.pad_token]] * (self.max_len - len(tokens))

        tokens = torch.tensor(tokens)
        predicate_tensor = torch.tensor(predicate_tensor)
        arg_labels = torch.tensor(arg_labels)

        # return tokens, predicate_tensor, torch.tensor(tokens.shape[0] * [0]), arg_labels
        return tokens, predicate_tensor, arg_labels

    def __test(self, idx):
        token_list = self.token_list[idx]
        predicate_start_idx, predicate_end_idx = self.predicate_idx[idx]
        arg_label_list = self.arg_label_list[idx]
        sentence = self.tokenizer.tokenizer_wordpiece.convert_tokens_to_string(token_list)
        word_list = sentence.split(" ")

        # ##################################################################################################
        tokens = []
        for t in token_list:
            if t in self.t2i:
                tokens.append(self.t2i[t])
            else:
                tokens.append(self.t2i[self.unknown_token])

        predicate_tensor = [0 for _ in range(self.max_len)]
        for i in range(predicate_start_idx, predicate_end_idx + 1):
            predicate_tensor[i] = 1


        arg_labels = []
        for word in word_list:
            if not arg_label_list:
                arg_labels.append(self.label_l2i["O"])
                continue

            label = arg_label_list[0]

            s_idx = label["start_idx"]
            e_idx = label["end_idx"]
            type = self.label_l2i[label["type"]]

            sample_token = self.tokenizer.tokenizer_wordpiece.convert_tokens_to_string(token_list[s_idx: e_idx + 1])

            if word == sample_token:
                del arg_label_list[0]
                arg_labels.append(type)
            else:
                arg_labels.append(self.label_l2i["O"])

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        elif len(tokens) < self.max_len:
            tokens = tokens + [self.t2i[self.pad_token]] * (self.max_len - len(tokens))

        for i in range(len(arg_labels), self.max_len):
            arg_labels.append(self.pad_id)

        tokens = torch.tensor(tokens)
        predicate_tensor = torch.tensor(predicate_tensor)
        arg_labels = torch.tensor(arg_labels)

        return tokens, predicate_tensor, arg_labels

if __name__ == "__main__":
    dataset = SrlDataset(
        vocab_path="./vocab",
        tokenizer_type="wordpiece",
        cache_path="./cache",
        data_path="data/ai_hub")

    print(len(dataset))
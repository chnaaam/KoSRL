from eunjeon import Mecab
from transformers import BertTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer_wordpiece = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer_mecab = Mecab()

    def tokenize(self, sentence, type):
        if type == "wordpiece":
            return self.tokenizer_wordpiece.tokenize(sentence)

        elif type == "mecab":
            return self.tokenizer_mecab.pos(sentence)

        else:
            raise ModuleNotFoundError()

    def __call__(self, sentence, predicate_info, args_info):

        # data = {
        #     'text': '인도양에 면해 있으며 북동쪽으로 소말리아, 북쪽으로 에티오피아와 남수단, ' +
        #             '서쪽으로 우간다, 남쪽으로 탄자니아와 국경을 맞닿고 있다.',
        #     'verb_arg_pair': {
        #         'verb': {'text': '맞닿고', 'idx': 13},
        #         'args': [
        #             {'type': 'ARGM-DIR', 'text': '서쪽으로', 'idx': 8},
        #             {'type': 'ARG2', 'text': '탄자니아와', 'idx': 11},
        #             {'type': 'ARG1', 'text': '국경을', 'idx': 12},
        #             {'type': 'AUX', 'text': '있다.', 'idx': 14}
        #         ]}}

        # Wordpiece


        return self.tokenize_wordpiece(sentence, predicate_info, args_info)

    def tokenize_wordpiece(self, sentence, predicate_info, args_info):

        predicate_start_idx, predicate_end_idx = -1, -1
        token_list = []
        space_idx_list = []
        arg_label_list = []

        predicate_info.setdefault("type", "PREDICATE")
        arg_list = [predicate_info]
        arg_list += args_info
        ordered_arg_list = {}
        for arg in arg_list:
            ordered_arg_list.setdefault(int(arg["idx"]), {"type": arg["type"], "text": arg["text"]})

        arg_list = sorted(ordered_arg_list.items())

        for word_idx, word in enumerate(sentence.split(" ")):
            tokens = self.tokenize(word, type="wordpiece")
            space_idx_list.append(len(tokens))

            if not arg_list:
                token_list += tokens
                continue

            arg = arg_list[0]

            arg_idx = arg[0]
            arg_data = arg[1]

            if arg_idx != word_idx:
                token_list += tokens
                continue

            del arg_list[0]

            start_idx = len(token_list)
            end_idx = len(token_list + tokens) - 1

            if arg_data["type"] == "PREDICATE":
                predicate_start_idx = start_idx
                predicate_end_idx = end_idx
            else:
                arg_label_list.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "type": arg_data["type"]})

            token_list += tokens

        return token_list, (predicate_start_idx, predicate_end_idx), arg_label_list


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer(data=None)
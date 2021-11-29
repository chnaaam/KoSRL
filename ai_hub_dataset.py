import os
import json

class AiHubDataset:
    def __init__(self, data_path):
        self.dataset = self.load_datasets(data_path)

    def load_datasets(self, data_path):
        srl_buffer = []
        file_list = os.listdir(data_path)

        for file_idx, file in enumerate(file_list):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as fp:
                data = json.load(fp)

            for d in data["sentence"]:
                text = d["text"]
                words = d["word"]
                srls = d["SRL"]

                if srls:
                    for srl in srls:
                        # 'verb': '면하',
                        # ... ,
                        # 'word_id': 1,
                        # ... ,
                        # 'argument': [{
                        #       'type': 'ARG2',
                        #       'word_id': 0,
                        #       'text': '인도양에',
                        #       'weight': 0.234661}, ]

                        word_idx = srl["word_id"]
                        verb = words[word_idx]["text"]

                        arg_list = []
                        for argument in srl["argument"]:

                            arg_list.append({
                                "type": argument["type"],
                                "text": words[argument["word_id"]]["text"],
                                "idx": argument["word_id"]
                            })

                        if arg_list:
                            srl_buffer.append({
                                "text": text,
                                "verb_arg_pair": {
                                    "verb": {"text": verb, "idx": word_idx},
                                    "args": arg_list}})

        return srl_buffer

if __name__ == "__main__":
    dataset = AiHubDataset(data_path="./data/ai_hub")
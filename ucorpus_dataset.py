import os


class UCorpusDataset:
    def __init__(
            self,
            data_path=None,
            data_fn=None):

        super(UCorpusDataset, self).__init__()

        self.data_path = data_path
        self.data_fn = data_fn
        self.data = self.load_corpus()

        # for debugging
        # for d in self.data:
        #     print(d)

    def load_corpus(self):
        corpus = []
        with open(os.path.join(self.data_path, self.data_fn), "r", encoding="utf-8") as fp:
            line_buffer = []

            for line in fp.readlines():
                line = line.replace("\n", "")

                if not line:
                    if line_buffer:
                        corpus.append(self.refine_corpus(line_buffer))
                        line_buffer = []
                else:
                    line_buffer.append(line)

        return corpus

    def refine_corpus(self, corpus):
        # corpus = """나를 본 사내가 반긴다.
        # 나__03/NP+를/JKO 보__01/VV+ㄴ/ETM 사내__01/NNG+가/JKS 반기/VV+ㄴ다/EF+./SF
        # #2	2 보__01	4 반기
        # 1 2 나__03/NP+를/JKO	THM
        # 2 3 보__01/VV+ㄴ/ETM
        # 3 4 사내__01/NNG+가/JKS		AGT
        # 4 4 반기/VV+ㄴ다/EF+./SF		"""

        sentence = corpus[0]
        word_list = sentence.split(" ")
        predicate_list = corpus[2].split("\t")
        predicate_num = int(predicate_list[0][1:])
        predicate_list = predicate_list[1:]

        for idx, c in enumerate(corpus[3:]):
            idx += 3

            c_list = c.split("\t")
            cs = c_list[0].split(" ")
            corpus[idx] = [cs[0], word_list[int(c[0]) - 1]] + c_list[1:]

        for idx, predicate in enumerate(predicate_list):
            corpus_idx = int(predicate.split(" ")[0]) + 2  # predicate index
            corpus[corpus_idx][idx + 2] = "PREDICATE"

        print(sentence)
        print(predicate_list)
        print()

        return {
            "sentence": sentence,
            "SRL": corpus[3:]}


if __name__ == '__main__':
    dataset = UCorpusDataset(
        data_path="data/ucorpus",
        data_fn="test_ucorpus.txt")

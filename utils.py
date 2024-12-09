import torch
import pickle as pkl
import pandas as pd

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_dataset(config):
    tokenizer = lambda x: [y for y in str(x)]  # char-level
    with open(config.vocab_path,'rb') as file:
        vocab = pkl.load(file)

    def load_dataset(path, pad_size=32):
        contents = []
        df = pd.read_excel(path)
        if "描述" not in df.columns:
            raise ValueError("文件中必须包含 '描述' 列！")
        descriptions = df["描述"].tolist()
        descriptions = [str(x) for x in descriptions]
        for content in descriptions:
            if not content:
                raise ValueError("'描述'列中描述不能为空！")
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, seq_len))
        return contents  # [([...], 4), ([...], 2), ...]
    test_data = load_dataset(config.test_path, config.pad_size)
    return test_data

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return (x, seq_len)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter
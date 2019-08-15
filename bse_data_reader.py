import numpy as np
import torch
from torch.utils.data import Dataset

from nltk.corpus import wordnet
import random

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, antonymFileName, min_count):

        self.negatives = []
        self.antonyms = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.antonymFileName = antonymFileName
        self.read_words(min_count)
        self.buildAntonymDictionary()
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        #for line in open(self.inputFileName, encoding="utf8"):
        for line in open(self.inputFileName):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        #word = word.strip('.,()[]{}?!;:-')
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        #for w, c in word_frequency.items():
        word_frequency_list = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        for w, c in word_frequency_list:
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def buildAntonymDictionary(self):
        f = open(self.antonymFileName, 'r')
        self.antonym_dict={}
        for line in f:
            tokens = line.strip('\n').split()
            if tokens[0] in self.word2id:
                self.antonym_dict[tokens[0]] = tokens[1:]    

    def getAntonym(self, target):
            antonyms = self.antonym_dict.get(self.id2word[target])
            if antonyms:
                return [self.word2id.get(random.choice(antonyms), target)]
            else:
                return [target]

# -----------------------------------------------------------------------------------------------------------------
class Word2VecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        #self.input_file = open(data.inputFileName, encoding="utf8")
        self.input_file = open(data.inputFileName)

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()
                #for word in words:
                    #word = word.strip('.,()[]{}?!;:-')

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


class BSEDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        #self.input_file = open(data.inputFileName, encoding="utf8")
        self.input_file = open(data.inputFileName)

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()
                #for word in words:
                    #word = word.strip('.,()[]{}?!;:-')

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, a, self.data.getNegatives(v, 5), self.data.getNegatives(a,5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v
                                for a in self.data.getAntonym(u)]
                   # return [(u, a, self.data.getNegatives(a, 5)) for i, u in enumerate(word_ids) 
                   #     for j, a in enumerate(self.antonyms(u,2*boundary))]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _, _ in batch if len(batch) > 0]
        all_a = [a for batch in batches for _, _, a, _, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, _, neg_v, _ in batch if len(batch) > 0]
        all_neg_a = [neg_a for batch in batches for _, _, _, _, neg_a in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_a), torch.LongTensor(all_neg_v), torch.LongTensor(all_neg_a)

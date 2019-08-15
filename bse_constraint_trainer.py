import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_reader import DataReader, Word2vecDataset
from model import SkipGramModel
import random


class Word2VecTrainer:
    def __init__(self, input_file, antonym_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12):

        print("Reading input file...")
        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        print("Creating data batches")
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)
        self.antonym_file = open(antonym_file,'r')

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def calculate_antonym_loss(self):

        src_ids = []
        tgt_ids = []

        while len(src_ids) < self.batch_size:
            line = self.antonym_file.readline()
            if not line:
                #EOF reached
                self.antonym_file.seek(0)
            words = line.strip('\n').split()
            if len(words) < 2:
                continue
            src=words[0]
            tgt=random.choice(words[1:]).strip('\n')

            src_id = self.data.word2id.get(src,None)
            tgt_id = self.data.word2id.get(tgt,None)

            if src_id is None or tgt_id is None:
                continue
    
            src_ids.append(src_id)
            tgt_ids.append(tgt_id)

        #src_embedding = self.skip_gram_model.embed(torch.LongTensor(src_id).to(self.device))
        #tgt_embedding = self.skip_gram_model.embed(torch.LongTensor(tgt_id).to(self.device))

        input_src = torch.LongTensor(src_ids).to(self.device)
        input_tgt = torch.LongTensor(tgt_ids).to(self.device)
        src_embedding = torch.squeeze(self.skip_gram_model.embed(input_src))
        tgt_embedding = torch.squeeze(self.skip_gram_model.embed(input_tgt))

        #loss = torch.abs(torch.dot(src_embedding,tgt_embedding))
        loss = torch.abs(torch.sum(torch.mul(src_embedding,tgt_embedding), dim=1))
        loss = loss/(torch.norm(src_embedding, dim=1)*torch.norm(tgt_embedding,dim=1))
        return torch.mean(loss)

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            count = 0

            for i, sample_batched in enumerate(self.dataloader):

                count += 1
                if count%10000 == 0:
                    print("\n\nEpoch %d, %d batches processed" % (iteration, count))

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    
                    skip_gram_loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    antonym_loss = 100*self.calculate_antonym_loss()

                    loss = skip_gram_loss + antonym_loss

                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50000 == 0:
                        print(" Loss: " + str(running_loss) + ' sk: ' + str(skip_gram_loss.data) + ' ant: ' + str(antonym_loss.data))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    #w2v = Word2VecTrainer(input_file="data/Wikipedia_first_10000_lines.txt", antonym_file='data/antonyms_wikipedia.txt', output_file="bse_constraint_vectors_window_5/out_iterations_3_cleaned.vec", iterations=3, min_count=50, window_size=5, emb_dimension=100)
    w2v = Word2VecTrainer(input_file="data/Wikipedia_text_with_periods_clean.txt", antonym_file='data/antonyms_wikipedia.txt', output_file="bse_constraint_vectors_window_5/out_iterations_3_cleaned.vec", iterations=3, min_count=50, window_size=5, emb_dimension=100)
    w2v.train()
    print('\nDone')

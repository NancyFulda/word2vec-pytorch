import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_reader import DataReader, Word2vecDataset
from model import SkipGramModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12):

        print("Reading input file...")
        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        print("Creating data batches")
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

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
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    #w2v = Word2VecTrainer(input_file="input.txt", output_file="embeddings.vec")
    #w2v = Word2VecTrainer(input_file="data/Wikipedia_first_50000000_lines.txt", output_file="vectors_DIM_50_window_5_small_corpus/out_iterations_3.vec", iterations=3, min_count=50, window_size=5, emb_dimension=50)
    #w2v = Word2VecTrainer(input_file="data/Wikipedia_text_with_periods_clean.txt",output_file="out_3_cleaned.vec", iterations=3, min_count=50)
    w2v = Word2VecTrainer(input_file="data/Wikipedia_text_with_periods_clean.txt", output_file="vectors_window_2/out_iterations_10_cleaned.vec", iterations=10, min_count=50, window_size=2, emb_dimension=100)
    w2v.train()
    print('\nDone')

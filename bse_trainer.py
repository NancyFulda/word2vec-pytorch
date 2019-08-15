import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from bse_data_reader import DataReader, BSEDataset, Word2VecDataset
from bse_model import BSEModel
import numpy as np

# modified from the original file from GitHub repo at https://github.com/Andras7/word2vec-pytorch

# Key change: in addition to treating words that appear nearby in the training corpus
# as each word's context, we also introduce a second training task: predicting the
# word's antonyms. The final word vectors are then taken from a combination of
# all three sets of weights.

NOHUP='w2v_nohup_window_5/nohup_iterations_3.out'

class BSETrainer:
    def __init__(self, input_file, antonym_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12):

        print("Reading input file...")
        self.data = DataReader(input_file, antonym_file, min_count)
        dataset = BSEDataset(self.data, window_size)
        print("Creating data batches")
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.bse_model = BSEModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.bse_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            with open(NOHUP, 'a') as f:
                f.write("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.bse_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            count = 0

            for sample_batched in self.dataloader:
            
                count += 1
                if count%10000 == 0:
                    print("\n\nEpoch %d, %d batches processed" % (iteration, count))

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    pos_a = sample_batched[2].to(self.device)
                    neg_v = sample_batched[3].to(self.device)
                    neg_a = sample_batched[4].to(self.device)

                loss = 0
                loss += self.bse_model.w2v_forward(pos_u, pos_v, neg_v)
                #loss += self.bse_model.ant_forward(pos_u, pos_a, neg_a)
                
                if loss != 0:
                    scheduler.step()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if count > 0 and count % 1000 == 0:
                        print(" Loss: ", str(loss.data)," running loss: ", str(running_loss))
                    
                    if count > 0 and count % 5000 == 0:
                        w2v_out = self.bse_model.w2v_sample(pos_u[0])
                        ant_out = self.bse_model.ant_sample(pos_u[0])

                        w2v_out = w2v_out.cpu().data.numpy()
                        ant_out = ant_out.cpu().data.numpy()

                        w2v_indices = np.where(w2v_out > 0.5)
                        ant_indices = np.where(ant_out > 0.5)
    
                        print('\n\ninput: ', self.data.id2word.get(pos_u.cpu().data.numpy()[0],'ERROR'))
                        
                        #print(ant_indices[0])
                        #print(w2v_indices[0])
                        #raw_input('>')

                        print('*** w2v words ***')
                        w2v_list = []
                        for element in w2v_indices[0]:
                            w2v_list.append(self.data.id2word.get(element,'ERROR'))
                        print(str(len(w2v_list)))
                        print(' '.join(w2v_list)+'\n')
                        print('*** ant words ***')
                        ant_list = []
                        for element in ant_indices[0]:
                            ant_list.append(self.data.id2word.get(element,'ERROR'))
                        print(str(len(ant_list)))
                        print(' '.join(ant_list)+'\n')

            self.bse_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    #bse = BSETrainer(input_file="data/Wikipedia_first_50000000_lines.txt", antonym_file='data/antonyms_wikipedia.txt',output_file="bse_vectors_DIM_50_window_5_small_corpus/out_iterations_5.vec", iterations=5, min_count=50, window_size=5, emb_dimension=50)
    #bse = BSETrainer(input_file="data/Wikipedia_text_with_periods.txt", antonym_file='data/antonyms_wikipedia.txt', output_file="junk.vec", iterations=5, min_count=50, window_size=5, emb_dimension=50)
    bse = BSETrainer(input_file="data/Wikipedia_first_10000_lines.txt", antonym_file='data/antonyms_wikipedia.txt',output_file="w2v_vectors_window_5/out_iterations_3_cleaned.vec", iterations=3, min_count=50, window_size=5, emb_dimension=100)
    #bse = BSETrainer(input_file="data/Wikipedia_text_with_periods_clean.txt", antonym_file='data/antonyms_wikipedia.txt',output_file="junk.vec", iterations=5, min_count=50, window_size=3, emb_dimension=100)
    #bse = BSETrainer(input_file="data/Wikipedia_first_10000_lines.txt", antonym_file='data/antonyms_wikipedia.txt',output_file="junk.vec", iterations=5, min_count=50, window_size=3, emb_dimension=100)
    bse.train()
    print('\nDone')
    with open(NOHUP, 'a') as f:
        f.write("\n\nDone")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
    a_embedding: Embedding for antonyms.
"""


class BSEModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(BSEModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.a_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        init.constant_(self.a_embeddings.weight.data, 0)

    def w2v_forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        
        return torch.mean(score + neg_score)

    def w2v_sample(self, pos_u):
        emb_u = self.u_embeddings(pos_u)
        scores = torch.matmul(emb_u,torch.t(self.v_embeddings.weight))
        scores = torch.clamp(scores, max=10, min=-10)
        scores = -F.logsigmoid(scores)
        return scores

    def ant_forward(self, pos_u, pos_a, neg_a):
        #context defined in terms of antonyms
        emb_u = self.u_embeddings(pos_u)
        emb_a = self.a_embeddings(pos_a)
        emb_neg_a = self.a_embeddings(neg_a)

        score = torch.sum(torch.mul(emb_u, emb_a), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_a, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def ant_sample(self, pos_u):
        emb_u = self.u_embeddings(pos_u)
        scores = torch.matmul(emb_u,torch.t(self.a_embeddings.weight))
        scores = torch.clamp(scores, max=10, min=-10)
        scores = -F.logsigmoid(scores)
        return scores

    def save_embedding(self, id2word, file_name):

        #save 'IN' embeddings
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name + '.IN', 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

        #save 'OUT' embeddings
        embedding = self.v_embeddings.weight.cpu().data.numpy()
        with open(file_name + '.OUT', 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
        
        #save 'ANT' embeddings
        embedding = self.a_embeddings.weight.cpu().data.numpy()
        with open(file_name + '.ANT', 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
        
        #save trio embeddings
        u_embedding = self.u_embeddings.weight.cpu().data.numpy()
        v_embedding = self.v_embeddings.weight.cpu().data.numpy()
        a_embedding = self.a_embeddings.weight.cpu().data.numpy()
        with open(file_name + '.DUAL', 'w') as f:
            f.write('%d %d\n' % (len(id2word), 3*self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), np.hstack([np.array(u_embedding[wid]), np.array(v_embedding[wid]), np.array(a_embedding[wid])])))
                f.write('%s %s\n' % (w, e))
        
        #save add embeddings
        embedding = (self.u_embeddings.weight + self.v_embeddings.weight + self.a_embeddings.weight).cpu().data.numpy()
        with open(file_name + '.ADD', 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

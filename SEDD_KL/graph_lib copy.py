import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)

class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        pass

    @abc.abstractmethod
    def rate(self, i):
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        pass

    def sample_transition(self, i, sigma):
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        pass

class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = torch.clamp(1 / esigm1.expand_as(x)[rel_ind], min=1e-8, max=1e8)
        other_ind = x0[rel_ind]
        score_rel = score[rel_ind]

        neg_term = ratio * score_rel.gather(-1, other_ind.unsqueeze(-1)).squeeze(-1)
        pos_term = torch.clamp(score_rel[:, :-1].exp().sum(dim=-1), min=1e-8, max=1e8)
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros_like(x, dtype=torch.float, device=x.device)
        entropy[rel_ind] = pos_term - neg_term + const
        return entropy

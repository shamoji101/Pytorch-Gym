
import torch
from torch import nn
from torch.nn import functional as F



"""
Seq2Seq

This code was built from this paper
https://arxiv.org/abs/1409.3215
"Sequence to Sequence Learning with Neural Networks"




"""



class Seq2Seq(nn.Module):

    """
    
    
    """

    def __init__(self, seq_len, input_size, hidden_size):
        
        super(Seq2Seq, self).__init__()
    
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        

    def forward(self, x):
        return x
        
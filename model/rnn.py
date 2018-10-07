import torch
import torch.nn as nn
import torch.nn.functional as F
 
class RNN_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN_Cell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.FC_hidden = nn.Linear(hidden_size, hidden_size)
        self.FC_input = nn.Linear(input_size, hidden_size)
        
        
    def forward(self, input_x, hidden):
        
        input_1=self.FC_hidden(hidden)
        input_2=self.FC_input(input_x)

        hidden = input_1 + input_2
        hidden = F.tanh(hidden)
        
        return hidden

        
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


class LSTM_Cell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM_Cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forgetunit_input_FC_x = nn.Linear(input_size, hidden_size)
        self.forgetunit_input_FC_h = nn.Linear(hidden_size, hidden_size)
        
        self.inputunit_input_FC_x = nn.Linear(input_size, hidden_size)
        self.inputunit_input_FC_h = nn.Linear(hidden_size, hidden_size)
        
        self.updateunit_input_FC_x = nn.Linear(input_size, hidden_size)
        self.updateunit_input_FC_h = nn.Linear(hidden_size, hidden_size)
        
        self.outputunit_input_FC_x = nn.Linear(input_size, hidden_size)
        self.outputunit_input_FC_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_x, hidden, cell):
        
        input_1=self.forgetunit_input_FC_x(input_x)
        input_2=self.forgetunit_input_FC_h(hidden)
        forget_x = input_1 + input_2
        forget_x = F.sigmoid(forget_x)
        cell_forget = cell * forget_x

        input_1=self.inputunit_input_FC_x(input_x)
        input_2=self.inputunit_input_FC_h(hidden)

        input_x = input_1 + input_2
        input_x = F.sigmoid(input_x)

        input_1=self.updateunit_input_FC_x(input_x)
        input_2=self.updateunit_input_FC_h(hidden)
        
        update_x = input_1 + input_2
        update_x = F.tanh(update_x)

        new_information = input_x * update_x
        cell_next = cell_forget + new_information

        input_1=self.outputunit_input_FC_x(input_x)
        input_2=self.outputunit_input_FC_h(hidden)
        
        output_x = input_1 + input_2
        output_x = F.sigmoid(output_x)
        
        hidden_next = output_x * F.tanh(cell_next)


        return hidden_next,cell_next
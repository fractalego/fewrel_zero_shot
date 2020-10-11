import torch.nn as nn
import torch.nn.functional as F


class RelTaggerModel(nn.Module):
    _bert_hidden_size = 768

    def __init__(self, language_model, ninp=200, dropout=0.2):
        super().__init__()
        self.language_model = language_model
        self.model_type = 'BERTREL'
        self.dropout = dropout

        self.input_linear = nn.Linear(self._bert_hidden_size, ninp)

        nout = 2
        self.linear_out1 = nn.Linear(ninp, nout)
        self.linear_out2 = nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.linear_out1.weight.data.uniform_(-initrange, initrange)
        self.linear_out2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, lengths):
        output = self.language_model(src)[0]

        output = self.input_linear(output)
        output = F.relu(output)

        out1 = self.linear_out1(output)
        out2 = self.linear_out2(output)

        subj_start, subj_end = [F.softmax(item[lengths[0]:].transpose(0, 1), dim=-1)
                                for item in out1.transpose(0, 2)]

        obj_start, obj_end = [F.softmax(item[lengths[0]:].transpose(0, 1), dim=-1)
                              for item in out2.transpose(0, 2)]

        return subj_start, subj_end, obj_start, obj_end

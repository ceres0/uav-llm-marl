import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from utils.th_utils import orthogonal_init_


class NRNNAgent(nn.Module):
    """PyMARL2 NRNNAgent supporting optional LayerNorm and orthogonal initialization."""

    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args
        self.rnn_hidden_dim = getattr(args, "rnn_hidden_dim", getattr(args, "hidden_dim", 64))

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(self.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=getattr(args, "gain", 0.01))

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if inputs.dim() == 3:
            b, a, e = inputs.size()
            flat_inputs = inputs.reshape(-1, e)
            reshape_3d = True
        else:
            flat_inputs = inputs
            reshape_3d = False

        x = F.relu(self.fc1(flat_inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        if reshape_3d:
            return q.view(b, a, -1), hh.view(b, a, -1)
        return q, hh

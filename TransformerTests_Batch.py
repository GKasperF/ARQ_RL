import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(torch.nn.Module):
  def __init__(self, out_size, ninp, nhead, nhid, nlayers, dropout=0.5):
    super(TransformerModel, self).__init__()
    try:
      from torch.nn import TransformerEncoder, TransformerEncoderLayer
    except Exception as e:
      raise ImportError('TransformerEncoder is not implemented')

    self.model_type = 'Transformer'
    self.src_mask = None
    self.pos_encoder = PositionalEncoding(ninp, dropout)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.ninp = ninp
    self.decoder = torch.nn.Linear(ninp, out_size)

    self.init_weights()

  def _generate_square_subsequent_mask(self, sz):
      mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
      mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
      return mask

  def init_weights(self):
      initrange = 0.1
      torch.nn.init.uniform_(self.encoder.weight, -initrange, initrange)
      torch.nn.init.zeros_(self.decoder.bias)
      torch.nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
  def forward(self, src, has_mask = True):
    if has_mask:
        device = src.device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
    else:
        self.src_mask = None

    src = self.pos_encoder(src)
    output = self.transformer_encoder(src, self.src_mask)
    output = self.decoder(output)
    return output


class ChannelModel(torch.nn.Module):
  def __init__(self, hidden_size, num_layers, output_size):
    super(ChannelModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.Layer1 = TransformerModel(out_size = Tf, ninp = 2, nhead = 5, nhid = hidden_size, nlayers = num_layers, dropout = 0.1)
    self.FinalLayer = torch.nn.Linear(hidden_size, output_size)
    self.prob_layer = torch.nn.Sigmoid()

  def forward(self, x, h, c):
    L1, (h_out, c_out) = self.Layer1(x, (h, c))
    L2 = self.FinalLayer(L1)
    output = self.prob_layer(L2)
    #output = torch.sigmoid(L2)

    return output, (h_out, c_out)

hidden_size = 10
num_layers = 5
batch_size = 1000
Tf = 10

RNN_Model = ChannelModel(hidden_size = hidden_size, num_layers = num_layers, output_size = Tf).to(device)
#criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.BCELoss(reduction='mean')
Params_LSTM = list(RNN_Model.Layer1.parameters())
Params_Linear = list(RNN_Model.FinalLayer.parameters())
optimizer = torch.optim.Adam(Params_LSTM + Params_Linear)
UpdateSteps = 1

with open('Data/Iid_Sequence_Example.pickle', 'rb') as f:
  Channel_Sequence_All = torch.load(f).to(device)
  # Channel_Sequence_All = Channel_Sequence_All[:, 0:1000]
  # Channel_Sequence_All = Channel_Sequence_All.repeat(1, 1)
# with open('Data/TraceSets/TraceUFSC_Failures.pth', 'rb') as f:
#   Channel_Sequence_All = torch.load(f).to(device)
#   Channel_Sequence_All = Channel_Sequence_All.repeat(610)
#   Channel_Sequence_All = Channel_Sequence_All[0:10000000].reshape(10000, 1000)

#Num_Samples = Channel_Sequence_All.shape[1]
Num_Samples = int( Channel_Sequence_All.shape[0]*Channel_Sequence_All.shape[1] / batch_size)
Channel_Sequence_All = Channel_Sequence_All.reshape(batch_size, Num_Samples).to(device)

update_count = 0
optimizer.zero_grad()
j=0
save_loss = np.zeros(int(((Num_Samples - Tf))/1)+1)

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)

temp = 0.5
prob_trans = temp*torch.ones(batch_size).to(device)
for i in range(Num_Samples - Tf):
    ChannelModel.train()
    target = Channel_Sequence_All[:, i + 1 : i + 1 + Tf]
    transmission = torch.bernoulli(prob_trans).type(torch.uint8)
    state_in = torch.cat( ((Channel_Sequence_All[:, i].type(torch.uint8) & transmission).type(torch.float).reshape((batch_size, 1, 1)), transmission.type(torch.float).reshape((batch_size, 1, 1))), dim=2)
    estimate, (h_out, c_out) = RNN_Model(state_in, h_in, c_in)
    loss = criterion(estimate, target.detach().reshape(estimate.shape))
    loss.backward()
    save_loss[i] = loss.detach().to('cpu').numpy()
    h_in = h_out.detach()
    c_in = c_out.detach()
    if (update_count % UpdateSteps == 0):
        optimizer.step()
        optimizer.zero_grad()
        j+=1
        print(update_count, j)
    update_count = update_count + 1

with open('Data/Iid_Loss_Example.pickle', 'wb') as f:
  torch.save(save_loss, f)

#with open('Data/RNN_Model_GE_Isolated_Erasures_Batch.pickle', 'wb') as f:
with open('Data/Iid_Model_Example.pickle', 'wb') as f:
  torch.save(RNN_Model, f)


save_loss = save_loss[:-1]
plt.plot(range(len(save_loss)), save_loss)
plt.xlabel('Step')
plt.ylabel('BCE Loss')
plt.show()

RNN_Model.Layer1.proj_size = 0
RNN_Model = RNN_Model.to(device)

batch_size = 1

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)


state_in0 = torch.tensor([0.0, 0.0]).to(device).reshape((batch_size, 1, 2))
state_in1 = torch.tensor([1.0, 1.0]).to(device).reshape((batch_size, 1, 2))
state_in2 = torch.tensor([0.0, 1.0]).to(device).reshape((batch_size, 1, 2))

estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
print('Starting erasure burst')
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
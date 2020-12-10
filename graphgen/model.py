import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .transformer_model import *

class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        ).cuda()

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        ).cuda()

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        ).cuda()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class RNN(nn.Module):
    """
    Custom GRU layer
    :param input_size: Size of input vector
    :param embedding_size: Embedding layer size (finally this size is input to RNN)
    :param hidden_size: Size of hidden state of vector
    :param num_layers: No. of RNN layers
    :param rnn_type: Currently only GRU and LSTM supported
    :param dropout: Dropout probability for dropout layers between rnn layers
    :param output_size: If provided, a MLP softmax is run on hidden state with output of size 'output_size'
    :param output_embedding_size: If provided, the MLP softmax middle layer is of this size, else 
        middle layer size is same as 'embedding size'
    :param device: torch device to instanstiate the hidden state on right device
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers, rnn_type='GRU',
        dropout=0, output_size=None, output_embedding_size=None,
        device=torch.device('cpu')
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_size = output_size
        self.device = device

        self.input = nn.Linear(input_size, embedding_size)

#         if self.rnn_type == 'GRU':
#             self.rnn = nn.GRU(
#                 input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
#                 batch_first=True, dropout=dropout
#             )
#         elif self.rnn_type == 'LSTM':
#             self.rnn = nn.LSTM(
#                 input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
#                 batch_first=True, dropout=dropout
#             )
        self.rnn = BartModel(BartConfig(decoder_attention_heads=8, num_hidden_layers=2, encoder_attention_heads=8, decoder_layers=4, encoder_layers=4, d_model=96, max_position_embeddings=256))
        self.rnn.cuda()
        # self.relu = nn.ReLU()

        self.hidden = None  # Need initialization before forward run

        if self.output_size is not None:
            if output_embedding_size is None:
                self.output = MLP_Softmax(
                    hidden_size, embedding_size, self.output_size).cuda()
            else:
                self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size).cuda()

#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.25)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(
#                     param, gain=nn.init.calculate_gain('sigmoid'))

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(
#                     m.weight.data, gain=nn.init.calculate_gain('relu'))

#     def init_hidden(self, batch_size):
#         if self.rnn_type == 'GRU':
#             # h0
#             return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
#         elif self.rnn_type == 'LSTM':
#             # (h0, c0)
#             return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
#                     torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
    def reset(self, batch_size):
        self.t = 0
        self.input_ids = torch.zeros((batch_size, 256, 96)).cuda()
        self.attention_mask = torch.zeros((batch_size, 256)).cuda()
        self.decoder_input_ids = torch.zeros((batch_size, 256, 96)).cuda()
        self.decoder_attention_mask = torch.zeros((batch_size, 256)).cuda()
        self.decoder_attention_mask[:, 0] = 1
        self.attention_mask[:, 0] = 1
        self.flag = False
#         self.attention_mask[:, 1] = 1
        self.past_key_values=None
        
        
    def forward(self, input, input_len=None):
        attention_mask = torch.zeros(input.shape[:-1]).cuda()
        attention_mask[:, 0] = 1
        input = self.input(input.cuda())
        # input = self.relu(input)
        if input_len is not None:
            self.decoder_input_ids = self.decoder_input_ids[:, :input_len.max()+1]
            self.decoder_attention_mask = self.decoder_attention_mask[:, :input_len.max()+1]
            outputs = torch.zeros_like(self.decoder_input_ids).cuda()
            attention_mask = torch.arange(input.shape[1]).view(1, -1).repeat(input.shape[0], 1).cuda() < input_len.unsqueeze(1)
            encoder_outputs=None
            past_key_values=None
            for i in range(input_len.max()):
#                 print(self.decoder_input_ids.shape, self.decoder_attention_mask.shape)
                tmp_out = self.rnn(input_ids=input, attention_mask = attention_mask, decoder_input_ids=self.decoder_input_ids.detach()[:, :self.t+1], decoder_attention_mask = self.decoder_attention_mask[:, :input_len.max()], encoder_outputs = encoder_outputs, past_key_values=past_key_values, use_cache=True)
                output = tmp_out['last_hidden_state']
                past_key_values = tmp_out['past_key_values']
                if encoder_outputs is None:
                    encoder_outputs = (tmp_out['encoder_last_hidden_state'], )
#                 print(output.shape, self.decoder_input_ids.shape)
                self.decoder_input_ids[:, self.t+1] = output.squeeze(1).detach()
                outputs[:, self.t+1] = output.squeeze(1)
                self.decoder_attention_mask[:, self.t+1] = 1
                self.t += 1
            if self.output_size is not None:
                outputs = self.output(outputs)
            return outputs[:, 1:input_len.max()+1]

#         if self.t == 0:
#             tmp_out = self.rnn(input_ids=self.input_ids, attention_mask = self.attention_mask, decoder_input_ids=self.decoder_input_ids, decoder_attention_mask = self.decoder_attention_mask, use_cache=False)
#             output = tmp_out['last_hidden_state']
#             self.decoder_input_ids[:, self.t] = output[:, self.t]
#             self.decoder_attention_mask[:, self.t] = 1
        
        self.input_ids[:, self.t] = input.squeeze(1)
        self.attention_mask[:, self.t] = 1
        tmp_out = self.rnn(input_ids=self.input_ids, attention_mask = self.attention_mask, decoder_input_ids=self.decoder_input_ids, decoder_attention_mask = self.decoder_attention_mask, use_cache=False)
        output = tmp_out['last_hidden_state'][:, self.t+1]
        
        if self.t < self.decoder_input_ids.shape[1]:
            self.decoder_input_ids[:, self.t+1] = output
            self.decoder_attention_mask[:, self.t+1] = 1
        self.t += 1
#         self.past_key_values = tmp_out['past_key_values']
        
        if self.output_size is not None:
            output = self.output(output)
#         print(output.max(), output.min(), self.t)
        return output.unsqueeze(1)


def create_model(args, feature_map):
    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1

    if args.note == 'DFScodeRNN':
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec
    else:
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec
    print("feature len", feature_len)
    if args.loss_type == 'BCE':
        MLP_layer = MLP_Softmax
    elif args.loss_type == 'NLL':
        MLP_layer = MLP_Log_Softmax

    dfs_code_rnn = RNN(
        input_size=feature_len, embedding_size=args.embedding_size_dfscode_rnn,
        hidden_size=args.hidden_size_dfscode_rnn, num_layers=args.num_layers,
        rnn_type=args.rnn_type, dropout=args.dfscode_rnn_dropout,
        device=args.device).to(device=args.device)

    output_timestamp1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_timestamp2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    model = {
        'dfs_code_rnn': dfs_code_rnn,
        'output_timestamp1': output_timestamp1,
        'output_timestamp2': output_timestamp2,
        'output_vertex1': output_vertex1,
        'output_vertex2': output_vertex2
    }

    if args.note == 'DFScodeRNN':
        output_edge = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_edge_output,
            output_size=len_edge_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)
        model['output_edge'] = output_edge

    return model

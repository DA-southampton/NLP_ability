import numpy as np
import torch
import torch.nn as nn

q1_numpy = np.load("q1_numpy.npy")
q1_lengths_numpy = np.load("q1_lengths_numpy.npy")
q2_numpy = np.load("q2_numpy.npy")
q2_lengths_numpy = np.load("q2_lengths_numpy.npy")


q1= torch.from_numpy(q1_numpy)
q1_lengths= torch.from_numpy(q1_lengths_numpy)
q2= torch.from_numpy(q2_numpy)
q2_lengths= torch.from_numpy(q2_lengths_numpy)


def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask		

q1_mask = get_mask(q1, q1_lengths)
q2_mask = get_mask(q2, q2_lengths)


q1_embed_numpy = np.load("q1_embed.npy")
q2_embed_numpy = np.load("q2_embed.npy")

q1_embed= torch.from_numpy(q1_embed_numpy)
q2_embed= torch.from_numpy(q2_embed_numpy)


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index



class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias, 
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
       
        
        reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs


first_rnn = Seq2SeqEncoder(nn.LSTM, 300, 300, bidirectional=True)


q1_encoded = first_rnn(q1_embed, q1_lengths)
#torch.Size([256, 29, 600])
q2_encoded = first_rnn(q2_embed, q2_lengths)
#torch.Size([256, 32, 600])


"""
import numpy as np
q1_encoded_numpy=q1_encoded.detach().numpy()
q2_encoded_numpy = q2_encoded.detach().numpy()

np.save("q1_encoded.npy", q1_encoded_numpy)
np.save("q2_encoded.npy", q2_encoded_numpy)



q1_encoded_numpy = np.load("q1_encoded.npy")
q2_encoded_numpy = np.load("q2_encoded.npy")

q1_encoded= torch.from_numpy(q1_encoded_numpy)
q2_encoded= torch.from_numpy(q2_encoded_numpy)
"""
## 计算attention

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):

    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


class SoftmaxAttention(nn.Module):

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)#prem_hyp_attn=[256,28,32]
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)##prem_hyp_attn=[256,32,28]
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses   

SoftmaxAttention=SoftmaxAttention()
q1_aligned, q2_aligned = SoftmaxAttention(q1_encoded, q1_mask, q2_encoded, q2_mask)


projection = nn.Sequential(nn.Linear(4*2*300, 300), nn.ReLU())


q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)


projected_q1 = projection(q1_combined)
projected_q2 = projection(q2_combined)

second_rnn = Seq2SeqEncoder(nn.LSTM, 300, 300, bidirectional=True)

q1_compare = second_rnn(projected_q1, q1_lengths)
q2_compare = second_rnn(projected_q2, q2_lengths)


def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add
               


q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)

merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
print(merged.shape)
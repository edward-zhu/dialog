import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from common.utils import pad_sequence

class SentenceEncoder(nn.Module):
    '''
        Encode a sentence to a vector

        param:
            vocab_size: vocabulary size
            word_dim: word vec dim
            hidden_size: GRU hidden size
            da: intermediate fc layer size
            output_dim: sentence vector size
            max_len: the max sentence length (in words)

        input: a sentence sequence of word indexes ([2, 4, 5, ...])
        output: an encoded vector 
    '''

    def __init__(self, embed, da, hidden_size, output_dim):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.da = da

        self.emb = embed
        self.gru = nn.GRU(input_size=self.emb.embedding_dim, 
                            hidden_size=self.hidden_size, bidirectional=True, dropout=True, batch_first=True)
        # self.fc = nn.Linear(2 * self.hidden_size, self.da)
        # self.att = nn.Linear(self.da, self.output_dim)

    def forward(self, sent):
        embds = self.emb(sent)  # b x n x d
        out, h = self.gru(embds)

        h = h.transpose(0, 1)
        # print h.size()

        return h.contiguous().view(sent.size(0), -1)

        # after_fc = F.tanh(self.fc(out)) # b * n * da
        # att = F.softmax(self.att(after_fc), dim=self.output_dim)
        # att_applied = att.transpose(1,2).bmm(out) # n * r * 2h

        # att_applied = att_applied.view(sent.size(0), -1) # flatten representation
        # final size b x 2h * output_dm
        # return att_applied

if __name__ == '__main__':
    import json
    from loader import load_data

    with open("config.json") as f:
        conf = json.load(f)

    # loader, embed = load_data(**conf)
    embed = nn.Embedding(20, 100)

    a = torch.LongTensor(pad_sequence([[2, 3, 4], [4, 5], [6, 7, 8, 9]]))
    enc = SentenceEncoder(embed, 14, 12, 2)
    a = Variable(a)
    y = enc(a)

    kb_found = torch.FloatTensor([1, 0, 1]).view(-1, 1)
    print torch.cat((y.data, kb_found), dim=1)

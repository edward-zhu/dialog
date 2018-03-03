import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

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

    def __init__(self, vocab_size, word_dim, da, hidden_size, output_dim, max_len):
        super(SentenceEncoder, self).__init__()
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.max_len = max_len
        self.da = da

        self.emb = nn.Embedding(vocab_size, word_dim)
        self.gru = nn.GRU(input_size=self.word_dim, 
                            hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * self.hidden_size, self.da)
        self.att = nn.Linear(self.da, self.output_dim)

    def forward(self, sent):
        embds = self.emb(sent)  # b x n x d
        out, h = self.gru(embds)
        after_fc = F.tanh(self.fc(out)) # b * n * da
        att = F.softmax(self.att(after_fc))
        att_applied = att.transpose(1,2).bmm(out) # n * r * 2h

        att_applied = att_applied.view(sent.size(0), -1) # flatten representation

        return att_applied

if __name__ == '__main__':
    a = torch.LongTensor([[2, 3, 4], [5, 6, 7]])
    enc = SentenceEncoder(10, 10, 14, 12, 1, 6)
    a = Variable(a)
    print enc(a)
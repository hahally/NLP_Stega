import torch
import torch.nn as nn

class BinsLstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super(BinsLstm, self).__init__()
    
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.word_embeding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.linear = nn.Linear(self.hidden_size, vocab_size)
    
    def init_hidden(self, bzs):
        h_0 = torch.zeros(self.num_layers, bzs, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, bzs, self.hidden_size)
        
        return (h_0, c_0)
    
    def forward(self, x, hidden):
        out = self.word_embeding(x)
        out, hidden = self.lstm(out, hidden)
        out = self.linear(out)
        
        return out, hidden
        
        
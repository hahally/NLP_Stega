import torch
from tqdm import tqdm
import argparse

from scripts.data_loader import create_dataloader
from scripts.utils import load_vocab, save_model
from models.model import BinsLstm

# train step function
def train_lstm_step(model, batch_data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(batch_data.size(0))
    hidden = (hidden[0].to(batch_data.device), 
              hidden[1].to(batch_data.device))
    seq_length = batch_data.size(1) - 1
    loss = 0
    
    for i in range(seq_length):
        out, hidden = model(batch_data[:,i].unsqueeze(dim=1), hidden)
        loss += criterion(out.squeeze(dim=1), batch_data[:,i+1])/seq_length
    loss.backward()
    optimizer.step()
    return loss

# eval model function
def evaluate(model, criterion, data_loader,device):
    loss = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            hidden = model.init_hidden(data.size(0))
            hidden = (hidden[0].to(device),
                      hidden[1].to(device))
            seq_length = data.size(1) - 1
            for i in range(seq_length):
                out, hidden = model(data[:,i].unsqueeze(dim=1), hidden)
                loss += criterion(out.squeeze(dim=1), data[:,i+1])/seq_length
    loss = loss/len(data_loader)
          
    return loss

# train function
def train(model, data, device, criterion, optimizer, epoch, print_freq, eval_freq, model_file):
    train_loader = data['train']
    valid_loader = data['valid']
    for e in range(epoch):
        print_info = {
            'Epoch':e,
            'step':0,
            'loss':0,
        }
        for step, data in enumerate(train_loader):
            data = data.to(device)
            loss = train_lstm_step(model, data, criterion, optimizer)
            # loss = loss/len(data)
            print_info['step'] = step + 1
            print_info['loss'] = loss.item()
            if (step+1) % print_freq==0:
                print(print_info)
                
            if (step+1) % eval_freq == 0:
                valid_loss = evaluate(model, criterion, valid_loader,device)
                print_info['loss'] = valid_loss.item()
                print(print_info)
                
            break
        
    # 验证
    valid_loss = evaluate(model, criterion, valid_loader,device)
    print_info['loss'] = valid_loss.item()
    print(print_info)
        
    # 保存模型
    save_model(model, model_file)
    print(f'saved model to {model_file}')
    
def main(args):
    # parameter configs
    epoch = args.epoch
    lr = args.lr
    print_freq = args.print_freq
    eval_freq = args.eval_freq
    batch_size = args.batch_size
    train_file = args.train_file
    valid_file = args.dev_file
    vocab_file = args.vocab_file
    model_file = args.model_file
    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    
    # data configs
    word2idx, idx2word = load_vocab(vocab_file)
    train_data = create_dataloader(word2idx,train_file,clip_len=50,batch_size=batch_size,shuffle=True)
    valid_data = create_dataloader(word2idx,valid_file,clip_len=50,batch_size=batch_size*10,shuffle=False)
    data_source = {
        'train': train_data,
        'valid': valid_data,
        # 'test': test_data
    }
    
    # model configs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinsLstm(vocab_size=len(word2idx),
                     embedding_dim=embedding_dim,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     dropout=dropout).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.98),
                                 eps=1e-9,
                                 weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])

    # train model
    train(model,data_source,device,criterion,optimizer,epoch,print_freq,eval_freq,model_file)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model For Text Generation')
    
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--train_file', type=str, default='./data/penn/train.txt')
    parser.add_argument('--dev_file', type=str, default='./data/penn/valid.txt')
    parser.add_argument('--test_file', type=str, default='./data/penn/test.txt')
    parser.add_argument('--vocab_file', type=str, default='./data/penn/vocab.pkl')
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Every xxx steps print log information.')
    parser.add_argument('--model_file', type=str, default="saved_model/model.pth")
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

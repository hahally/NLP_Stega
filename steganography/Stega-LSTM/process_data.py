from scripts.utils import create_vocab, load_vocab, save_vocab


if __name__ == '__main__':
    file_list = ['./data/penn/train.txt','./data/penn/valid.txt']
    save_path = './data/penn/vocab.pkl'
    word2index, index2word = create_vocab(file_list=file_list,vocab_num=25000)
    save_vocab(word2index, index2word, save_path)

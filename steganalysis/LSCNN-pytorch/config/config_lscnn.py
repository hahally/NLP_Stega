class Config():
    # train files
    cover_file = './data/quora-train-src.txt'
    stego_file = './data/round_1.txt'
    
    # test files
    test_files = ['./data/test.txt']
    
    # corpus files
    corpus = ['./data/quora-train-src.txt', './data/round_1.txt']
    
    train_model = True
    test_model = False
    
    # model
    lr_mode = 'dynamic' # dynamic, static, both
    embedding_dim = 300
    num_classes = 2
    kernel_size = [3,5,7]
    kernel_num = 128
    drop_rate = 0.5
    w2v_path = ''
    save_model_path = './saved_model/best.model'
    device = 'cuda'
    
    # train
    lr = 0.001
    batch_size = 128
    epoch = 30
    early_stop = 30
    seed = 2023
    works = 0
    print_freq = 100
    
    # token
    pad = 0
    unk = 1
    
    

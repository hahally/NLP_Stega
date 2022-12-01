import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from LS_CNN.models import LScnn
import os

random.seed(202)
np.random.seed(2022)
tf.random.set_seed(2022)


def train_model(train,
                test,
                tokens,
                w2v_emb,
                k_folds=5,
                mode='both',
                model_path='',
                lr_rate=0.001, 
                batch_size=16, 
                expoch=200, 
                max_len=32, 
                num_classes=2,
                random_state=2022):
    
    lr = lr_rate
    batch_size = batch_size
    max_len = max_len
    num_classes = num_classes
    train = train
    mode = mode
    expoch = expoch
    random_state = random_state
    
    tokens = tokens
    
    print(f'train data:[{len(train)}]')
    
    word2indx = tokens.word_index
    vocab_size = len(word2indx) + 1

    x_train = tokens.texts_to_sequences(train.sentence.tolist())
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len,padding='post',truncating='post')
    y_train = train.label.to_numpy()
    
    
    x_test = tokens.texts_to_sequences(test.sentence.tolist())
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    y_test = test.label.to_numpy()
    
    folds = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    oof = np.zeros([len(train), num_classes])
    predictions = np.zeros([len(test), num_classes])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        print("fold n{}".format(fold_+1))

        model = LScnn(w2v_emb,
                      num_classes=num_classes,
                      max_len=max_len,
                      lr_mode=mode,
                      vocab_size=vocab_size,
                      embed_size=300,
                      kernel_sizes=[3, 5, 7],
                      kernel_num=128,
                      drop_rate=0.5
                      )
        
        bst_model_path = os.path.join(model_path, f"{mode}.{fold_+1}.h5")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      metrics=['accuracy'])
        
        X_tra, X_val = x_train[trn_idx], x_train[val_idx]
        y_tra, y_val = y_train[trn_idx], y_train[val_idx]
        
        X_tra = tf.constant(X_tra)
        y_tra = tf.constant(y_tra)
        X_val = tf.constant(X_val)
        y_val = tf.constant(y_val)
        model.fit(X_tra, y_tra,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val),
                  epochs=expoch,
                  callbacks=[early_stopping, model_checkpoint]
                  )
        
        model.load_weights(bst_model_path)
        oof[val_idx] = model.predict(X_val)
        predictions += model.predict(x_test) / folds.n_splits

    pre = np.argmax(predictions, axis = -1)
    average = 'binary'
    if num_classes>2:
        average = 'micro'
    
    accuracy = (pre == y_test).sum()/len(pre)
    f1_s = f1_score(y_test, pre, average=average)
    precision = precision_score(y_test, pre, average=average)
    recall = recall_score(y_test, pre, average=average)
    print(f'On train data: accuracy:{accuracy},f1:{f1_s},precision:{precision},recall:{recall}')
    cm = confusion_matrix(y_test, pre)
    print('cm:',cm)
    return accuracy, f1_s, precision, recall, cm
    

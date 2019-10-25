# -*- coding: utf-8 -*-
"""

@author: dongdong1994
"""

from data_process import DataProcess
from data_generate import generate_batch
from encoder2decoder import build_model

from test import data_to_padding_ids
from test import predict_text
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
def run():
    batch_size = 63
    epochs = 5000
    
    data_process = DataProcess(use_word2cut=False)

    model = build_model()
  
    documents_length = data_process.get_documents_size(data_process.enc_ids_file, data_process.dec_ids_file)
    
    if batch_size > documents_length:
        print("ERROR--->" + u"语料数据量过少，请再添加一些")
        return None
    #自适应学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-6, mode='min')
    '''monitor: 需要监视的量，val_loss，val_acc
    patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
    verbose: 信息展示模式
    mode: 'auto','min','max'之一，在min模式训练，如果检测值停止下降则终止训练。在max模式下，当检测值不再上升的时候则停止训练。'''
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    model.fit_generator(generator=generate_batch(batch_size=batch_size),
                        steps_per_epoch=int(documents_length / batch_size)+5, \
                        validation_data=generate_batch(batch_size=batch_size), \
                        validation_steps=int(documents_length / batch_size)+5,\
                        epochs=epochs, verbose=1, workers=2, use_multiprocessing=True,
                        callbacks=[reduce_lr,early_stopping])

    model.save_weights("model/seq2seq_model_weights.h5", overwrite=True)
    
if __name__ == "__main__":
    run()
    
    
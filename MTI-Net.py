"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Layer, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.activations import softmax
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, CuDNNLSTM
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import argparse
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import pdb
random.seed(999)

epoch=10
batch_size=1
forgetgate_bias=-3

NUM_EandN=15000
NUM_Clean=1500


def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def Sp_and_phase(path, Noisy=False):
    
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal)) 
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
      
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    end2end = np.reshape(signal,(1,signal.shape[0],1))
    return NLp, end2end

def data_generator(file_list, file_list_ssl, list_stoi):
	index=0
	while True:
         wer_filepath=file_list[index].split(',')
         wav2vec_filepath=file_list_ssl[index].split(',')
         stoi_filepath=list_stoi[index].split(',')
         
         noisy_LP,noisy_end2end =Sp_and_phase(wer_filepath[3]) 
         
         noisy_ssl =np.load(wav2vec_filepath[0])          
         wer=np.asarray(float(wer_filepath[2])).reshape([1])
         intell=np.asarray(float(wer_filepath[1])).reshape([1])
         stoi=np.asarray(float(stoi_filepath[1])).reshape([1])

         feat_length_end2end = math.ceil(float(noisy_end2end.shape[1])/256)
         final_len = noisy_LP.shape[1] + int(feat_length_end2end) + noisy_ssl.shape[1]
         
         index += 1
         if index == len(file_list):
             index = 0
            
             random.Random(7).shuffle(file_list)
             random.Random(7).shuffle(file_list_ssl)

         yield  [noisy_LP, noisy_end2end, noisy_ssl], [wer, wer[0]*np.ones([1,final_len,1]), intell, intell[0]*np.ones([1,final_len,1]), stoi, stoi[0]*np.ones([1,final_len,1])]

def BLSTM_CNN_with_ATT_cross_domain():
    input_size =(None,1)
    _input = Input(shape=(None, 257))
    _input_end2end = Input(shape=(None, 1))

    SincNet_ = Sinc_Conv_Layer(input_size, N_filt = 257, Filt_dim = 251, fs = 16000, NAME = "SincNet_1").compute_output(_input_end2end)
    merge_input = concatenate([_input, SincNet_],axis=1) 
    re_input = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input)
        
    # CNN
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
    re_shape = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
    _input_ssl = Input(shape=(None, 768))
    bottleneck=TimeDistributed(Dense(512, activation='relu'))(_input_ssl)
    concat_with_ssl = concatenate([re_shape, bottleneck],axis=1) 
    blstm=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_ssl)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)
    
    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention')(dense1)
    Frame_score=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score')(attention)
    WER_score=GlobalAveragePooling1D(name='WER_score')(Frame_score)
    
    attention2 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-6),bias_regularizer=keras.regularizers.l1(1e-6),attention_regularizer_weight=1e-6, name='Attention2')(dense1)
    Frame_intell=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_intell')(attention2)
    Intell_score= GlobalAveragePooling1D(name='Intell_score')(Frame_intell)
    
    attention3 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-6),bias_regularizer=keras.regularizers.l1(1e-6),attention_regularizer_weight=1e-6, name='Attention3')(dense1)
    Frame_stoi=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_stoi')(attention3)
    STOI_score= GlobalAveragePooling1D(name='STOI_score')(Frame_stoi)
    
    model = Model(outputs=[WER_score, Frame_score, Intell_score , Frame_intell, STOI_score , Frame_stoi], inputs=[_input,_input_end2end, _input_ssl])
  
    return model

def train(train_list, train_list_ssl, train_list_STOI, NUM_TRAIN, valid_list, valid_list_ssl, valid_list_STOI, NUM_VALID, pathmodel):    
    print ('model building...')
    model = BLSTM_CNN_with_ATT_cross_domain()
    adam = Adam(lr=1e-5)
    model.compile(loss={'WER_score': 'mse', 'Frame_score': 'mse', 'Intell_score': 'mse', 'Frame_intell': 'mse', 'STOI_score': 'mse', 'Frame_stoi': 'mse'}, optimizer=adam)
    plot_model(model, to_file=pathmodel+'.png', show_shapes=True)
    
    with open(pathmodel+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min')  
    
    print ('training...')
    g1 = data_generator(train_list, train_list_ssl, train_list_STOI)
    g2 = data_generator(valid_list, valid_list_ssl, valid_list_STOI)

    hist=model.fit_generator(g1,steps_per_epoch=NUM_TRAIN, epochs=epoch, verbose=1,validation_data=g2,validation_steps=NUM_VALID,max_queue_size=1, workers=1,callbacks=[checkpointer])
               					
    model.save(pathmodel+'.h5')

    # plotting the learning curve
    TrainERR=hist.history['loss']
    ValidERR=hist.history['val_loss']
    print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
    print 'drawing the training process...'
    plt.figure(2)
    plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
    plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
    plt.xlim([1,epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig('Learning_curve_'+pathmodel+'.png', dpi=150)


def test(Test_list_QI_score, Test_list_ssl, Test_list_wer_score,pathmodel):
    print 'testing...'
    model = BLSTM_CNN_with_ATT_cross_domain()
    model.load_weights(pathmodel+'.h5')

    WER_Predict=np.zeros([len(Test_list_QI_score),])
    WER_true   =np.zeros([len(Test_list_QI_score),])

    Intell_Predict=np.zeros([len(Test_list_QI_score),])
    Intell_true   =np.zeros([len(Test_list_QI_score),])

    for i in range(len(Test_list_QI_score)):
       print i
       Asessment_filepath=Test_list_QI_score[i].split(',')
       noisy_LP, noisy_end2end =Sp_and_phase(Asessment_filepath[2]) 
       noisy_ssl = np.load(Test_list_ssl[i])
    
       path = Test_list_wer_score[i].split(',')
       wer = float(path[0])
       intell=float(Asessment_filepath[1])

       [WER_1, frame_score, Intell_1, frame_intell, STOI_1, frame_stoi]=model.predict([noisy_LP,noisy_end2end,noisy_ssl], verbose=0, batch_size=batch_size)

       WER_Predict[i]=WER_1
       WER_true[i]   =wer

       Intell_Predict[i]=Intell_1
       Intell_true[i]   =intell

    MSE=np.mean((WER_true-WER_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(WER_true, WER_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(WER_true.T, WER_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(WER_Predict),1])
    plt.figure(1)
    plt.scatter(WER_true, WER_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True WER')
    plt.ylabel('Predicted WER')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_WER_'+pathmodel+'.png', dpi=150)

    MSE=np.mean((Intell_true-Intell_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(Intell_true, Intell_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(Intell_true.T, Intell_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(Intell_Predict),1])
    plt.figure(2)
    plt.scatter(Intell_true, Intell_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True Intell')
    plt.ylabel('Predicted Intell')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_Intell_'+pathmodel+'.png', dpi=150)


if __name__ == '__main__':  
     
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0') 
    parser.add_argument('--mode', type=str, default='train') 
    
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel="MTI-Net_Cross_Domain"

    #################################################################             
    ######################### Training data #########################
    ###  LSTM Enhanced ###
    Train_list_wav = ListRead('/Lists/TMHINT_QI_Train_with_Quality_Intelligibility_WER_label.txt')
    Train_list_STOI = ListRead('/Lists/TMHINT_QI_Train_with_PESQ(Norm)_STOI_label.txt')
    Train_list_ssl = ListRead('/Lists/List_SSL_Train.txt')
    NUM_DATA =  len(Train_list_wav)

    NUM_TRAIN = int(NUM_DATA*0.9) 
    NUM_VALID = NUM_DATA-NUM_TRAIN

    train_list= Train_list_wav[: NUM_TRAIN]
    random.Random(7).shuffle(train_list)
    valid_list= Train_list_wav[NUM_TRAIN: ]

    train_list_ssl= Train_list_ssl[: NUM_TRAIN]
    random.Random(7).shuffle(train_list_ssl)
    valid_list_ssl= Train_list_ssl[NUM_TRAIN: ]

    train_list_STOI = Train_list_STOI[: NUM_TRAIN]
    random.Random(7).shuffle(train_list_STOI)
    valid_list_STOI= Train_list_STOI[NUM_TRAIN: ]

    ################################################################
    ######################### Testing data #########################
    Test_list_QI_score=ListRead('/Lists/TMHINT_QI_Test_with_Quality_Intelligibility_label.txt')
    Test_list_wer_score=ListRead('/Lists/TMHINT_QI_Test_with_CER_label.txt')    
    Test_list_ssl=ListRead('/Lists/List_SSL_Test.txt')
    Num_testdata= len (Test_list_QI_score)

    if args.mode == 'train':
       print 'training'  
       train(train_list, train_list_ssl, train_list_STOI, NUM_TRAIN, valid_list, valid_list_ssl, valid_list_STOI, NUM_VALID, pathmodel)
       print 'complete training stage'    
    
    if args.mode == 'test':      
       print 'testing' 
       test(Test_list_QI_score, Test_list_ssl, Test_list_wer_score,pathmodel)
       print 'complete testing stage'
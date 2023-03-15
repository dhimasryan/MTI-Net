"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from FT_SSL_Feat import MosPredictor, MyDataset
import numpy as np
import scipy.stats
import torchaudio
import pdb
import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')     
    parser.add_argument('--filename', type=str, default='List_SSL_Train')       
    args = parser.parse_args()
    
    cp_path = '/pre-trained_model/wav2vec_small.pt'
    my_checkpoint = '/FT_Model/ckpt_21' #path where we save the fine-tuned model checkpoint
     
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()
    model.load_state_dict(torch.load(my_checkpoint))

    Test_List=ListRead('/data/Lists/train_mos_list.txt')
  
    print('Starting prediction')
    list_new =[]   

    if args.mode =="train":
       directory ='/data/Train_SSL_Feat/'
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)  
    
    else :
       directory ='/data/Test_SSL_Feat/'    
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)               
          
    for i in range(len(Test_List)):   
        Asessment_filepath=Test_List[i].split(',')
        wavefile = Asessment_filepath[2]
        path_name = wavefile
        S=path_name.split('/')
        wave_name=S[-1]
        name = wave_name[:-4] 
        new_name =  name +'.npy' 
               
        cached_path = os.path.join(directory,new_name)
        wav = torchaudio.load(wavefile)[0] 
        wav = wav.to(device)
   
        WER_1,Intell_1,stoi,res = model(wav)
        res_feat = res['x']        
        res_feat = res_feat.detach().to("cpu").numpy()
        np.save(cached_path,res_feat)        
        info = str(cached_path)
        list_new.append(info)
    
    with open(args.filename+'.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item) 

if __name__ == '__main__':
    main()

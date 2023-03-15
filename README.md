# MTI-Net: A Multi-Target Speech Intelligibility Prediction Model

### Introduction ###

Recently, deep learning (DL)-based non-intrusive speech assessment models have attracted great attention. Many studies report that these DL-based models yield satisfactory assessment performance and good flexibility, but their performance in unseen environments remains a challenge. Furthermore, compared to quality scores, fewer studies elaborate deep learning models to estimate intelligibility scores. This study proposes a multi-task speech intelligibility prediction model, called MTI-Net, for simultaneously predicting human and machine intelligibility measures. Specifically, given a speech utterance, MTI-Net is designed to predict human subjective listening test results and word error rate (WER) scores. We also investigate several methods that can improve the prediction performance of MTI-Net. First, we compare different features (including low-level features and embeddings from self-supervised learning (SSL) models) and prediction targets of MTI-Net. Second, we explore the effect of transfer learning and multi-tasking learning on training MTI-Net. Finally, we examine the potential advantages of fine-tuning SSL embeddings. Experimental results demonstrate the effectiveness of using cross-domain features, multi-task learning, and fine-tuning SSL embeddings. Furthermore, it is confirmed that the intelligibility and WER scores predicted by MTI-Net are highly correlated with the ground-truth scores. 

For more detail please check our <a href="https://www.isca-speech.org/archive/pdfs/interspeech_2022/zezario22_interspeech.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MOSA-Net_Cross_Domain.py, Generate_PS_Feature.py, Generate_end2end_Feature.py```. To generate Self Supervised Learning (SSL) feature, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/pytorch/fairseq" target="_blank">link</a> to deploy fairseq module.  
### Feature Extaction ###

For extracting cross-domain features, please use Generate_end2end_Feature.py, Generate_PS_Feature.py, Generate_SSL_Feature.py. When extracting SSL feature, please make sure that fairseq can be imported correctly. Please refer to this link for detail <a href="https://github.com/pytorch/fairseq" target="_blank">installation</a>. 

Please follow the following format to make the input list.
```js
PESQ score, STOI score, SDI score, filepath directory
```

### How to run the code ###

Please use following script to train the model:
```js
python MOSA-Net_Cross_Domain.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MOSA-Net_Cross_Domain.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
Zezario, R.E., Fu, S.-w., Chen, F., Fuh, C.-S., Wang, H.-M., Tsao, Y. (2022) MTI-Net: A Multi-Target Speech Intelligibility Prediction Model. Proc. Interspeech 2022, 5463-5467

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/pytorch/fairseq" target="_blank">Self-Supervised Learning Model</a> are created by others

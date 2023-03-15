# MTI-Net: A Multi-Target Speech Intelligibility Prediction Model

### Introduction ###

Recently, deep learning (DL)-based non-intrusive speech assessment models have attracted great attention. Many studies report that these DL-based models yield satisfactory assessment performance and good flexibility, but their performance in unseen environments remains a challenge. Furthermore, compared to quality scores, fewer studies elaborate deep learning models to estimate intelligibility scores. This study proposes a multi-task speech intelligibility prediction model, called MTI-Net, for simultaneously predicting human and machine intelligibility measures. Specifically, given a speech utterance, MTI-Net is designed to predict human subjective listening test results and word error rate (WER) scores. We also investigate several methods that can improve the prediction performance of MTI-Net. First, we compare different features (including low-level features and embeddings from self-supervised learning (SSL) models) and prediction targets of MTI-Net. Second, we explore the effect of transfer learning and multi-tasking learning on training MTI-Net. Finally, we examine the potential advantages of fine-tuning SSL embeddings. Experimental results demonstrate the effectiveness of using cross-domain features, multi-task learning, and fine-tuning SSL embeddings. Furthermore, it is confirmed that the intelligibility and WER scores predicted by MTI-Net are highly correlated with the ground-truth scores. 

For more detail please check our <a href="https://www.isca-speech.org/archive/pdfs/interspeech_2022/zezario22_interspeech.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MTI-Net.py```. To generate and fine-tuned Self Supervised Learning (SSL) feature, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/pytorch/fairseq" target="_blank">link</a> to deploy fairseq module.  

### Fine-tuned SSL model and Extact SSL Feature ###

Please use the following code to fine-tuned SSL model:
```js
python FT_SSL_Feat.py
```
To extract the SSL feature, please use the following code:
```js
python Extract_FT_SSL.py
```

### Train and Testing MTI-Net ###

Please use following script to train the model:
```js
python MTI-Net.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MTI-Net.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
Zezario, R.E., Fu, S.-w., Chen, F., Fuh, C.-S., Wang, H.-M., Tsao, Y. (2022) MTI-Net: A Multi-Target Speech Intelligibility Prediction Model. Proc. Interspeech 2022, 5463-5467

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/pytorch/fairseq" target="_blank">Self-Supervised Learning Model</a> are created by others

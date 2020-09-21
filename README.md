# MasterProject
ECG Classification

## Overview

This project is going to classifiy 6-class Arrhythmia using SVM, Xgboost, CNN, LSTM. 

## Install

If you don't have wfdb，install it with

```
pip install wfdb
```

This is a package which is used for processing MIT-BIH database. 

If you don't have `Biosppy 0.6.1`, install it with 

```
pip install biosppy
```

which is a package for ECG signal processing. 

You can setup the environment using setup.sh

Make sure that the environment is `python 3.7`

## Dataset

You need to download the MIT-BIH dataset from [Physionet](https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip). Unzip and rename the folder is mitbd. Put the folder in Data. 

Download the Data, which is segment by an 1800 length of the sliding window, and the interval is 100.  [Google Drvie](https://drive.google.com/file/d/1uce2k-iDb8Kd1tAYrvQlX7Hd1hsU-xno/view?usp=sharing) Put the file in the folder which is named as Data. 

## Details

There are 5 index you can choose when you run `main.py`. 

### Preprocessing

* You can put the number 0 to run the Preprocessing function. 
* You will get the Heart Beats dataset for training and testing. 

### Training

****

#### Based on Heartbeats

##### SVM

* You can type the number 1 to run SVM. 
* Input: WT feature
* Output: Test results and report

##### Xgboost

* You can type the number 2 to run Xgboost. 
* Input: WT feature
* Output: Testing results and report

##### CNN

* You can type the number 3 to train and test CNN network. 
* Input: deniose heartbeats signal
* Output: Testing result and report

##### LSTM

* You can type the number4 to train and test LSTM network. 
* Input: deniose heartbeats signal
* Output: Testing result and report

#### Based on ECG Segments

##### LSTM

* You can type the number 5 to train and test LSTM network. 
* Input: Segment dataset (input size is 1800)
* Output: Testing result and report

##### GRU 

* You can type the number 6 to train and test GRU network. 
* Input: Segment dataset (input size is 1800)
* Output: Testing result and report

##### GRU with Attention

* You can type the number 7 to train and test GRU with attention network. 
* Input: Segment dataset (input size is 1800)
* Output: Testing result and report

#### Reproduce

##### 11  layers CNN network[1]

* You can type the number 8 to train and test GRU network. 
* Input: Segment dataset (input size is 1250)
* Output: Testing result and report

##### 16  layers CNN network[2]

* You can type the number 9 to train and test GRU network. 
* Input: Segment dataset (input size is 1800) (The inout of origninal paper is 3600 )
* Output: Testing result and report

##### CNN-LSTM

* You can type the number 10 to train and test GRU network. 
* Input: Segment dataset (input size is 1000)
* Output: Testing result and report

## Environment

```
biosppy == 0.6.1
torch == 1.4.0
wfdb == 2.2.1
xgboost == 1.0.2
scikit-learn == 0.23.1
scipy == 1.4.0
PyWavelets == 1.1.1
```



## Reference

**[1]** U. R. Acharya, H. Fujita, O. S. Lih, Y. Hagiwara, J. H. Tan, and M. Adam, \Automated detection of arrhythmias using dierent intervals of tachycardia ecg segments with convolutional neural network," Information sciences, vol. 405, pp. 81{90, 2017. 

**[2]**  O. Yldrm, P. P  lawiak, R.-S. Tan, and U. R. Acharya, \Arrhythmia detection using deep convolu- tional neural network with long duration ecg signals," Computers in biology and medicine, vol. 102, pp. 411{420, 2018. 

**[3]** S. L. Oh, E. Y. Ng, R. San Tan, and U. R. Acharya, \Automated diagnosis of arrhythmia using combination of cnn and lstm techniques with variable length heart b eats," Computers in biology and medicine, vol. 102, pp. 278{287, 2018

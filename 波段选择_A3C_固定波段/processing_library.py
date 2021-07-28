# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import copy
import os

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
# import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/data/' + data_name + '/' + data_name + '_pre.mat')
    
    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori']
#    x_train = pre['train_x']
    y_train = pre['train_y'][0]
    train_loc = pre['train_loc']
#    x_test = pre['test_x']
    y_test = pre['test_y'][0]
    test_loc = pre['test_loc']
    y_val = pre["val_y"][0]
    val_loc = pre["val_loc"]
    ulab_loc = pre['Ulab_loc']

    
    return data_norm,labels_ori,y_train,train_loc,y_test,test_loc,y_val,val_loc,ulab_loc

def one_hot(lable,class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array
###############################################################################
def disorder(X,Y):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    return X,Y

def disorder_unl(X):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    return X
###############################################################################
def next_batch(image,lable,index,batch_size):
    '''数据分批'''
    start = index-batch_size
    end = index
    return image[start:end,:],lable[start:end]
def next_batch_unl(image,index,batch_size):
    start = index-batch_size
    end = index
    return image[start:end,:]

def contrary_one_hot(label):
    '''将onehot标签转化为真实标签'''
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])
    return label_ori

###############################################################################
def index_band_selection(w):
    '''找到所选波段位置'''
    c=np.where(w!=0)[0].tolist()
    return c
###############################################################################
def save_result(data_name,oa,aa,kappa,num_band_seclection_now,band_loction,per_class_acc,train_time,test_time):
    '''将实验结果保存在txt文件中'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'num_band_seclection:'+str(num_band_seclection_now)+'\n'+'band_loction:'+str(band_loction)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt','a')
    f.writelines(write_content)
    f.close()
    return       

def expend(data, w):
    size = data.shape
    data_expand = np.zeros((int(size[0]+w-1), int(size[1]+w-1), size[2]))
    for j in range(size[2]):
        data_expand[:, :, j] = np.lib.pad(data[:, :, j], ((
            int(w / 2), int(w / 2)), (int(w / 2), int(w / 2))), 'symmetric')
    return data_expand


def windowFeature(data_expand, loc, w):
    newdata = np.zeros([loc.shape[0], w, w, data_expand.shape[2]])
    for i in range(loc.shape[0]):
        x1 = loc[i, 0]
        y1 = loc[i, 1]
        x2 = loc[i, 0] + w
        y2 = loc[i, 1] + w
        c = data_expand[x1:x2, y1:y2, :]
        newdata[i, :, :, :] = c
    return newdata


def pca_trans_expend(data, n, w):
    '''PCA + expend
    Args:
        data: input data, size like (W,H,b)
        n : n_components of PCA, a integer number
        w : width of patchszie, a odd number
    '''
    data_reshape = data.reshape((-1, data.shape[2]))
    pca = PCA(n_components=n)
    data_pca = pca.fit_transform(data_reshape)
    data_reshape_2 = data_pca.reshape([data.shape[0], data.shape[1], -1])
    data_ex = expend(data_reshape_2, w)
    return data_ex

def random_cut_bands(data):
    '''
    按均匀分布随机选择波段集合
    '''
    
#    num_bands = data.shape[1]
#    for i in range(data.shape[0]):
#        num_zero = num_bands - int(np.random.uniform(low=1, high=num_bands))
#        index_zero_bands = np.random.choice(range(num_bands),num_zero,replace=False)
#        data[i,index_zero_bands,:,:] = 0
    num_bands = data.shape[1]
    num_zero = num_bands - int(np.random.uniform(low=1, high=num_bands))
    index_zero_bands = np.random.choice(range(num_bands),num_zero,replace=False)
    data[:,index_zero_bands,:,:] = 0
    return data
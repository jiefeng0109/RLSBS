# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, default='PaviaU',
                    help='数据集名称')
parser.add_argument('--pt', type=float, default=0.05,
                    help='训练样本比例')
parser.add_argument('--pv', type=float, default=0.05,
                    help='无标签样本比例')
FLAGS, unparsed = parser.parse_known_args()

def readData(data_name):
    ''' 读取原始数据和标准类标 '''
    path = os.getcwd()+'/data/'+data_name
    if data_name == 'Indian_pines':
        data = sio.loadmat(
            path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == 'PaviaU':
        data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
        labels = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == 'KSC':
        data = sio.loadmat(path+'/KSC.mat')['KSC']
        labels = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']
    elif data_name == 'Salinas':
        data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
    elif data_name == 'washington':
        data = sio.loadmat(path+'/washington.mat')['washington_datax']
        labels = sio.loadmat(path+'/washington_gt.mat')['washington_labelx']
    elif data_name == 'Houston':
        data = sio.loadmat(path+'/Houstondata.mat')['Houstondata']
        labels = sio.loadmat(path+'/Houstonlabel.mat')['Houstonlabel']
    data = np.float64(data)
    labels = np.array(labels).astype(float)
    return data, labels


# def normalizeData(data):
#     ''' 原始数据归一化处理（每条） '''
#     data_norm = np.zeros(np.shape(data))
#     for i in range(np.shape(data)[0]):
#         for j in range(np.shape(data)[1]):
#             data_norm[i, j, :] = preprocessing.normalize(
#                 data[i, j, :].reshape(1, -1))[0]
#     return data_norm

def normalizeData(data):
    ''' 原始数据归一化处理（每个特征） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[2]):
        x = preprocessing.normalize(data[:,:,i].reshape([1,-1]))
        data_norm[:,:,i] = x.reshape([data.shape[0],data.shape[1]])
    return data_norm


def selectTrainTestVal(data, labels, pt, pv):
    ''' 从所有类中每类选取训练样本和测试样本
        pt:训练样本比例
        pv：验证样本比例 
     '''
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 训练样本
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])  # 验证
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []

    z = np.array([], dtype=float).reshape(-1, data.shape[2])  # 测试
    zb = []
    z_loc1 = []
    z_loc2 = []
    z_loc = []

    for i in range(1, c+1):
        #i = 1
        loc1, loc2 = np.where(labels == i)
        # print("isequal",loc1==loc2)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]

        num1 = int(np.round(num*(pt)))  # 训练
        num2 = int(np.round(num*(pt+pv)))  # 验证
        # print(num1,num2,num)
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])
        y = np.vstack([y, data[loc1[num1:num2], loc2[num1:num2], :]])
        z = np.vstack([z, data[loc1[num2:], loc2[num2:], :]])

        xb.extend([i]*num1)
        yb.extend([i]*(num2-num1))#caution
        zb.extend([i]*(num-num2))

        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])

        y_loc1.extend(loc1[num1:num2])
        y_loc2.extend(loc2[num1:num2])

        z_loc1.extend(loc1[num2:])
        z_loc2.extend(loc2[num2:])

        x_loc = np.vstack([x_loc1, x_loc2])
        y_loc = np.vstack([y_loc1, y_loc2])
        z_loc = np.vstack([z_loc1, z_loc2])

        x_loc = np.array(x_loc).transpose()
        y_loc = np.array(y_loc).transpose()
        z_loc = np.array(z_loc).transpose()
    return x, xb, x_loc, y, yb, y_loc,z,zb,z_loc 



def selectUlabelSamples(data, labels, pu):
    # x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 
    # x_loc1 = []
    # x_loc2 = []
    x_loc = []
    loc1, loc2 = np.where(labels == 0)
    num = len(loc1)
    print(num)
    order = np.random.permutation(range(num))
    loc1 = loc1[order]
    loc2 = loc2[order]
    num1 = int(np.round(num*(pu)))  # 
    print(num1)
    x = data[loc1[:num1], loc2[:num1], :]
    x_loc1= loc1[:num1]
    x_loc2= loc2[:num1]
    x_loc = np.vstack([x_loc1, x_loc2])
    x_loc = np.array(x_loc).transpose()
    return x,x_loc

if __name__ == '__main__':

    data_name = FLAGS.data_name
#    data_name = 'KSC'
#    data_name = 'Salinas'
#    data_name = 'PaviaU' 
 #   data_name = 'Houston'

    data_ori, labels_ori = readData(data_name)
    data_norm = normalizeData(data_ori)
    pt,pv=FLAGS.pt,FLAGS.pv
    pu= 0.01
    # if data_name == 'Indian_pines':
    #     pt,pv,pu = 0.05,0.05,0.1 #Train Val Ulabel S
    # elif data_name == 'PaviaU':
    #     pt,pv,pu = 0.03,0.03,0.1
    # elif data_name == 'KSC':
    #     pt,pv,pu =0.05,0.05,0.5
    # elif data_name == 'Salinas':
    #     pt,pv,pu = 0.05,0.05,0.5
    # elif data_name == 'washington':
    #     pt,pv,pu = 0.05,0.05,0.5
    # elif data_name == 'Houston':
    #     pt,pv,pu = 0.05,0.05,0.5
    train_x, train_y, train_loc,val_x,val_y,val_loc, test_x, test_y, test_loc = selectTrainTestVal(data_norm, labels_ori, pt,pv)
    Ulab_x, Ulab_loc = selectUlabelSamples(data_norm, labels_ori,pu)
    

    order = np.random.permutation(range(train_x.shape[0]))
    train_x = train_x[order]
    train_y = np.array(train_y)[order]
    train_loc = train_loc[order]

    path = os.getcwd()
    sio.savemat(path+'/data/'+data_name+'/'+data_name+'_pre.mat', {'train_x':train_x,
                'train_y':train_y, 'train_loc':train_loc,
                'val_x':val_x,
                'val_y':val_y, 'val_loc':val_loc,
                'test_x':test_x,
                'test_y':test_y, 'test_loc':test_loc, 'data_norm':data_norm,
                'Ulab_x':Ulab_x,'Ulab_loc':Ulab_loc,
                'labels_ori':labels_ori})
    print("Finished!")


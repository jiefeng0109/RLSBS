# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:53:52 2019

@author: ld

This is a environment for hyperspectral bandselection
达到停止条件时：reward = over all accuary
其他状态 ：reward = 0

这是一个半监督版本，同时输入有监督和半监督样本

停止条件：选中停止Action
"""

import numpy as np
from net import Net
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch, index_band_selection
from processing_library import contrary_one_hot, expend, windowFeature, pca_trans_expend
from processing_library import save_result, random_cut_bands

import copy
import math
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#_num_classification = 16


class HSIEnv:
    """
    ENV for band selection search
    停止条件：选中停止Action
    """

    def __init__(self, model, data, num_bands_all, max_num_bands_selection,
                 num_classification, reward_rare=300, lambada_cost=0,):
        '''
        data: a dict ["data_norm","X_valid_loc","Y_valid","w"]
        max_num_bands_selection: 选中的最大波段数目
        '''
        # save options
        self.model = model.cuda()  # 预训练过的评价网络
#        print(self.model)
        # self.checkpoint = checkpoint
        self.reward_rare = reward_rare
        self.data = data
        self.data_norm = torch.tensor(
            data["data_norm"], dtype=torch.float32).cuda()
        self.X_valid_loc = torch.tensor(
            data["X_valid_loc"], dtype=torch.int32).cuda()

        self.Y_valid = torch.tensor(data["Y_valid"], dtype=torch.long).cuda()
        self.unl_loc = torch.tensor(
            data["unl_loc"][:self.X_valid_loc.shape[0], :], dtype=torch.int32).cuda()  # !
        self.w = data["w"]
        # 将数据预加载到显存中
        self.X_valid = self._windowFeature_torch(self.data_norm, self.X_valid_loc, self.w).permute(
            0, 3, 1, 2)
        self.X_valid_unl = self._windowFeature_torch(self.data_norm, self.unl_loc, self.w).permute(
            0, 3, 1, 2)
        print(self.X_valid.shape,self.X_valid_unl.shape)
        # othe options
        self.num_bands_selection = max_num_bands_selection
        self.lambada_cost = lambada_cost
        self.num_bands_all = num_bands_all
        self.num_classification = num_classification
        # 已选波段集合
        self.selected_bands = np.zeros(
            [num_bands_all])  # one-hot形式，1代表选中，0代表未被选中

        # build reward

        self.reset()        #
        self.fitness_full = self._validate(
            self.data, self.model, torch.ones([num_bands_all]).cuda())
        print('=> all bands acc: {:.3f}'.format(self.fitness_full))
        self.reward = 0
        self.fitness = 0
        self.best_reward = -math.inf
        self.best_strategy = None

    def _count_slected_bands(self, selected_bands):
        '''统计已选波段数量'''
        return np.sum(selected_bands).astype(np.uint8)

#    def _cut_bands_by_index(self, data, index):
#        return np.where(data[:,index,:,:]==1,data,0)
#
    def _cut_bands_by_index(self, data, index):
        '''在GPU中完成，以实现加速'''
        data_new = torch.zeros_like(data)
        data_new[:, index, :, :] = data[:, index, :, :]
        return data_new

    def _one_hot(self, index, n_bands):
        # 波段！
        one_hot_array = np.zeros([n_bands])
        one_hot_array[int(index)] = 1
        return one_hot_array

    def _contrary_one_hot_torch(self, label):
        label_ori = torch.zeros([label.shape[0], ], dtype=torch.long).cuda()
        for i in range(label.shape[0]):
            label_ori[i] = torch.argmax(label[i])
        return label_ori

    def _windowFeature_torch(self, data_expand, loc, w):
        with torch.no_grad():
            newdata = torch.zeros(
                [loc.shape[0], w, w, data_expand.shape[2]]).cuda()
            for i in range(loc.shape[0]):
                x1 = loc[i, 0]  # 没错！
                y1 = loc[i, 1]
                x2 = loc[i, 0] + w
                y2 = loc[i, 1] + w
                c = data_expand[x1:x2, y1:y2, :]
                newdata[i, :, :, :] = c
            return newdata

    def _center_Loss_torch(self, feature, y_true, num_classification, selected_bands):
        # 有监督的centerloss
        # 2020年3月2日修正
        total_center = torch.mean(feature, axis=0, keepdim=True)
    #    print(feature,y_true)
    #    total_center_tile=total_center.view([1,1,feature.shape[1]]).repeat([feature.shape[0],num_classification,1])
    #    print(total_center_tile.shape)
        class_center = torch.tensor([], dtype=torch.float32).cuda().view(
            [-1, feature.shape[1]])  # 类内中心

        for i in range(num_classification):
            index = torch.where(y_true == torch.tensor(
                i, dtype=torch.long).cuda())[0]
    #        print("index",index)
            if index.shape[0] != 0:
                result = torch.mean(feature[index, :], axis=0, keepdim=True)
    #            print("1",result.shape)
            else:
                result = total_center
    #            print("2",result.shape)
            class_center = torch.cat([class_center, result])
    #        print("30",class_center.shape)
        '''有标签样本特征类内距离'''
        class_center_gather = class_center[y_true]
        loss_wtl = torch.mean(torch.div(torch.sum(torch.abs(
            feature-class_center_gather), axis=1, keepdim=True).view([feature.shape[0]]), selected_bands if selected_bands > 0 else 1))
        return loss_wtl

    def _center_Loss_us_torch(self, feature, y_pred_us, num_classification, selected_bands):
        # 无监督的centerloss
        # 2020年3月2日修正
        y_afs = F.softmax(y_pred_us, dim=-1)
    #    print(y_afs.shape)
    #    print(torch.argmax(y_afs,axis =1))
        y_afs_tile = y_afs.view([feature.shape[0], num_classification, 1]).repeat(
            [1, 1, feature.shape[1]])
        feature_unl_tile = feature.view(
            [feature.shape[0], 1, feature.shape[1]]).repeat([1, num_classification, 1])

        pij_sum = torch.sum(y_afs, axis=0, keepdim=True).view(
            [num_classification, 1]).repeat([1, feature.shape[1]])

        class_center_unl = torch.div(torch.sum(torch.mul(feature_unl_tile, y_afs_tile), axis=0, keepdim=True).view(
            [num_classification, feature.shape[1]]), pij_sum)

        class_center_unl_tile1 = (class_center_unl.view(
            [1, num_classification, feature.shape[1]]).repeat([feature.shape[0], 1, 1]))
        '''无标签样本特征类内距离，固定/不固定特征中心，没有太大差别，采用固定的方式loss_wtu1'''
        loss_wtu1 = torch.sum(torch.mul(torch.div(torch.sum(torch.abs(feature_unl_tile-class_center_unl_tile1),
                                                            axis=2).view([feature.shape[0], num_classification]), selected_bands if selected_bands > 0 else 1), y_afs))/(feature.shape[0])

        return loss_wtu1

    def _get_fitness(self, data_norm, X_valid, Y_valid, X_valid_unl, w, band_index):
        # 半监督版本
        size = np.shape(X_valid)
        num = size[0]
        index_all = 0
        step_ = 2000
        loss = 0

        while index_all < num:
            t0 = time.time()
            if index_all + step_ > num:
                input = X_valid[index_all:, :, :, :]
                input_unl = X_valid_unl[index_all:, :]
                batch_y = Y_valid[index_all:]
            else:
                input = X_valid[index_all:(index_all+step_), :, :, :]
                input_unl = X_valid_unl[index_all:(index_all+step_), :, :, :]
                batch_y = Y_valid[index_all:(index_all+step_)]
            t1 = time.time()
            x = input
            t2 = time.time()
            x_unl = input_unl
#            print(input_loc_unl.shape)
            # print(batch_y)
            t3 = time.time()
            batch_y = self._contrary_one_hot_torch(batch_y)

            input = self._cut_bands_by_index(x, band_index)
            t4 = time.time()
            input_unl = self._cut_bands_by_index(x_unl, band_index)
    #        input = input.reshape([input.shape[0],input.shape[1],input.shape[2],input.shape[3],1])
            index_all += step_
            t5 = time.time()
            # print(input.shape,input_unl.shape)
            temp1, featrue = self.model(input)
            t6 = time.time()
            temp2, featrue_unl = self.model(input_unl)
#            print(nn.CrossEntropyLoss()(temp1, batch_y).cpu())
            entropy = nn.CrossEntropyLoss()(temp1, batch_y).item()
            loss_ce = self._center_Loss_torch(
                featrue, batch_y, self.num_classification, np.sum(band_index))
            loss_ce_us = self._center_Loss_us_torch(
                featrue_unl, temp2, self.num_classification, np.sum(band_index))
            loss = loss + entropy + 1.5*loss_ce.item() + 1.5*loss_ce_us.item()
#            print("Fetch Data Time",t3-t0)
#            print("Cut Bands time",t5-t3)
#            print("Inference Time:",t6-t5)
        return loss

    def _validate(self, data, model, selected_bands, verbose=False):
        '''
        Validate the performance on validation set
        :param data:
        :param model:
        :param verbose:
        :return: over all accuary
        '''
        bands_index = self.bands_one_hot_2_index(selected_bands)
#        print(bands_index)
        t1 = time.time()
        with torch.no_grad():
            fitness = self._get_fitness(
                self.data_norm, self.X_valid, self.Y_valid, self.X_valid_unl, data["w"], bands_index)
#            oa = self._get_oa(
#                self.data_norm, self.X_valid_loc,self.Y_valid, data["w"], bands_index)
            t2 = time.time()
            self.fitness = fitness
            if verbose:
                print("fitness:%.3f     time:%.3f" % (fitness, t2-t1))

        return fitness

    def reset(self):
        self.s = np.zeros([self.num_bands_all])
        self.reward = 0
        return self.s

    def _visited(self, s, action):
        visited = False
        if np.max(s+action) > 1:
            visited = True
        return visited

    def cal_reward(self, s):
        fitness = self._validate(self.data, self.model, s)
        return (self.fitness_full-fitness)*self.reward_rare

    def step(self, action,episode):
        info = None
        done = False
        # action为停止条件时，计算loss，停止
        if action == self.num_bands_all:
            s_ = self.s  # 状态不变
            self.reward = self.cal_reward(s_)
            done = True
            info = self.fitness
            #print("STOPED!,selected bands: ",np.sum(self.s))
        else:
            action = self._one_hot(action, self.num_bands_all)
            if self._visited(self.s, action) and np.sum(self.s) < self.num_bands_selection:
                # 选中重复波段时，奖励-1
                s_ = self.s
    #            self.reward = self.cal_reward(self.s)-0.5
                self.reward = -0.1
                info = -1
            elif np.sum(self.s) < self.num_bands_selection - 1:
                # 选中非重复波段，奖励-lambda_cost
                s_ = self.s + action
                self.s = s_
    #            self.reward = self.cal_reward(self.s)
                self.reward = 0. - self.lambada_cost
                info = 1
            else:
                s_ = self.s + action
                self.reward = self.cal_reward(s_)
                done = True
                info = self.fitness
                #lambda递减
                # if episode>3000:
                #     self.lambada_cost = self.lambada_cost+.1/2000

    #            self.reward =  0
    #            done = True
    #            info = 0
    #            print(info)

        return s_, self.reward, done, info

    def bands_one_hot_2_index(self, selected_bands):
        bands_index = [i for i in range(
            len(selected_bands)) if selected_bands[i] == 1]
        return bands_index

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from Utils.minmax_normalizer import MinMaxNormalization, MinMaxNormalization_01
import time

def get_taxibj_data(flow_data,ext_data,n_closeness,n_period,n_trend,tt_split,g_closeness=1,g_period=24*2,g_trend=24*2*7):
    data_len, feature, map_height, map_width = flow_data.shape
    _, ext_dim = ext_data.shape
    assert ((n_closeness,n_period,n_trend) < (7,4,3))
    start_idx = g_trend*n_trend
    total_samples = data_len-start_idx
    n_train = int(total_samples*tt_split)
    n_test = total_samples-n_train
    
    tstamp_train = np.zeros(n_train)
    y_train = np.zeros((n_train,2,map_height,map_width))
    x_closeness_train = np.zeros((n_train,n_closeness,2,map_height,map_width))    
    x_period_train = np.zeros((n_train,n_period,2,map_height,map_width))
    x_trend_train = np.zeros((n_train,n_trend,2,map_height,map_width))
    x_ext_train = np.zeros((n_train,ext_dim))
    k = 0
    for i in range(start_idx,start_idx+n_train):    
        tstamp_train[k] = i
        y_train[k,:,:,:] = flow_data[i,:,:,:]
        x_ext_train[k,:] = ext_data[i,:]
        l = 0
        for j in range(i-g_trend, i-g_trend*(n_trend+1), -g_trend):
            x_trend_train[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1        
        l = 0
        for j in range(i-g_period, i-g_period*(n_period+1), -g_period):            
            x_period_train[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1
        l = 0
        for j in range(i-g_closeness, i-g_closeness*(n_closeness+1), -g_closeness):            
            x_closeness_train[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1
        k += 1    
    x_closeness_train = x_closeness_train.reshape(n_train,-1,map_height,map_width)
    x_period_train = x_period_train.reshape(n_train,-1,map_height,map_width)
    x_trend_train = x_trend_train.reshape(n_train,-1,map_height,map_width)
    
    start_idx += n_train
    tstamp_test = np.zeros(n_test)
    y_test = np.zeros((n_test,2,map_height,map_width))
    x_closeness_test = np.zeros((n_test,n_closeness,2,map_height,map_width))    
    x_period_test = np.zeros((n_test,n_period,2,map_height,map_width))
    x_trend_test = np.zeros((n_test,n_trend,2,map_height,map_width))
    x_ext_test = np.zeros((n_test,ext_dim))
    k = 0
    for i in range(start_idx,start_idx+n_test):
        tstamp_test[k] = i
        y_test[k,:,:,:] = flow_data[i,:,:,:]
        x_ext_test[k,:] = ext_data[i,:]
        l = 0
        for j in range(i-g_trend, i-g_trend*(n_trend+1), -g_trend):
            x_trend_test[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1        
        l = 0
        for j in range(i-g_period, i-g_period*(n_period+1), -g_period):            
            x_period_test[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1
        l = 0
        for j in range(i-g_closeness, i-g_closeness*(n_closeness+1), -g_closeness):            
            x_closeness_test[k,l,:,:,:] = flow_data[j,:,:,:]
            l += 1
        k += 1
    x_closeness_test = x_closeness_test.reshape(n_test,-1,map_height,map_width)
    x_period_test = x_period_test.reshape(n_test,-1,map_height,map_width)
    x_trend_test = x_trend_test.reshape(n_test,-1,map_height,map_width)
    
    return tstamp_train, x_closeness_train, x_period_train, x_trend_train, x_ext_train, y_train, \
           x_closeness_test, x_period_test, x_trend_test, x_ext_test, y_test

import torch
from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TaxiBJDataset(Dataset):
    def __init__(self,path,n_closeness,n_period,n_trend,tt_split,train=True):
        flow2013 = np.load(path+'TaxiBJ2013.npy')
        flow2014 = np.load(path+'TaxiBJ2014.npy')
        flow2015 = np.load(path+'TaxiBJ2015.npy')
        flow2016_1 = np.load(path+'TaxiBJ2016_1.npy')
        flow2016_2 = np.load(path+'TaxiBJ2016_2.npy')
        flow2016 = np.vstack((flow2016_1, flow2016_2))
        ext2013 = np.load(path+'TaxiBJext2013.npy')
        ext2014 = np.load(path+'TaxiBJext2014.npy')
        ext2015 = np.load(path+'TaxiBJext2015.npy')
        ext2016 = np.load(path+'TaxiBJext2016.npy')
        
        self.mmn_flow = MinMaxNormalization()
        self.mmn_flow.fit(np.vstack((flow2013, flow2014, flow2015, flow2016)))
        self.mmn_temp = MinMaxNormalization_01()
        self.mmn_temp.fit(np.hstack((ext2013[:,8], ext2014[:,8], ext2015[:,8], ext2016[:,8])))
        self.mmn_wind = MinMaxNormalization_01()
        self.mmn_wind.fit(np.hstack((ext2013[:,9], ext2014[:,9], ext2015[:,9], ext2016[:,9])))
        
        flow2013 = self.mmn_flow.transform(flow2013)
        flow2014 = self.mmn_flow.transform(flow2014)
        flow2015 = self.mmn_flow.transform(flow2015)
        flow2016 = self.mmn_flow.transform(flow2016)
        
        ext2013[:,8] = self.mmn_temp.transform(ext2013[:,8])[:]
        ext2014[:,8] = self.mmn_temp.transform(ext2014[:,8])[:]
        ext2015[:,8] = self.mmn_temp.transform(ext2015[:,8])[:]
        ext2016[:,8] = self.mmn_temp.transform(ext2016[:,8])[:]
        
        ext2013[:,9] = self.mmn_wind.transform(ext2013[:,9])[:]
        ext2014[:,9] = self.mmn_wind.transform(ext2014[:,9])[:]
        ext2015[:,9] = self.mmn_wind.transform(ext2015[:,9])[:]
        ext2016[:,9] = self.mmn_wind.transform(ext2016[:,9])[:]
        
        if train:
            _, self.x_c, self.x_p, self.x_t, self.x_e, self.y, _, _, _, _, _ = get_taxibj_data(flow2013,ext2013,n_closeness,n_period,n_trend,tt_split)
            _, x_c_, x_p_, x_t_, x_e_, y_, _, _, _, _, _ = get_taxibj_data(flow2014,ext2014,n_closeness,n_period,n_trend,tt_split)    
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
            _, x_c_, x_p_, x_t_, x_e_, y_, _, _, _, _, _ = get_taxibj_data(flow2015,ext2015,n_closeness,n_period,n_trend,tt_split)    
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
            _, x_c_, x_p_, x_t_, x_e_, y_, _, _, _, _, _ = get_taxibj_data(flow2016,ext2016,n_closeness,n_period,n_trend,tt_split)    
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
        else:
            _, _, _, _, _, _, self.x_c, self.x_p, self.x_t, self.x_e, self.y = get_taxibj_data(flow2013,ext2013,n_closeness,n_period,n_trend,tt_split)
            _, _, _, _, _, _, x_c_, x_p_, x_t_, x_e_, y_ = get_taxibj_data(flow2014,ext2014,n_closeness,n_period,n_trend,tt_split)
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
            _, _, _, _, _, _, x_c_, x_p_, x_t_, x_e_, y_ = get_taxibj_data(flow2015,ext2015,n_closeness,n_period,n_trend,tt_split)
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
            _, _, _, _, _, _, x_c_, x_p_, x_t_, x_e_, y_ = get_taxibj_data(flow2016,ext2016,n_closeness,n_period,n_trend,tt_split)
            self.stack_data(x_c_, x_p_, x_t_, x_e_, y_)
        self.dataset_len = self.y.shape[0]
        self.y = torch.tensor(self.y, device=torch.device(device)).float()
        self.x_c = torch.tensor(self.x_c, device=torch.device(device)).float()
        self.x_p = torch.tensor(self.x_p, device=torch.device(device)).float()
        self.x_t = torch.tensor(self.x_t, device=torch.device(device)).float()
        self.x_e = torch.tensor(self.x_e, device=torch.device(device)).float()
        # print(self.x_c.shape, self.x_p.shape, self.x_t.shape, self.x_e.shape) 

        
    def stack_data(self, x_c_, x_p_, x_t_, x_e_, y_):
        self.x_c = np.vstack((self.x_c, x_c_))
        self.x_p = np.vstack((self.x_p, x_p_))
        self.x_t = np.vstack((self.x_t, x_t_))
        self.x_e = np.vstack((self.x_e, x_e_))
        self.y = np.vstack((self.y, y_))
        
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.x_c[idx], self.x_p[idx], self.x_t[idx], self.x_e[idx], self.y[idx]
    
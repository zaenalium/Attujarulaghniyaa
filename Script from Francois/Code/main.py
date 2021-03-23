from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from random import randint

import os
cwd = os.path.abspath(os.path.curdir) 
rundttm = datetime.today().strftime('%y%m%d_%H%M')
fpath = cwd + '\\' + rundttm + '\\'
os.makedirs(fpath)
flogname = fpath + 'RunLog.txt' 
flog = open(flogname, 'a')

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import itemfreq

np.set_printoptions(threshold=np.nan, suppress=True)
pd.set_option('display.width', 1000)

import charts as chrt
import classifiers as clsf
import patterns as ptrn
import transform as trns

def load_data(stock, lob_level, format):
    
    t1 = time()
    
    msgfile = cwd + '\\data\\' + stock + '_Messages_' + repr(lob_level) + '.csv' 
    obfile = cwd + '\\data\\' + stock + '_OB_' + repr(lob_level) + '.csv' 
    
    df1=pd.read_csv(msgfile, sep=',', header=None, 
            names=['time', 'type', 'ord_id', 'size', 'price', 'dir'],
            dtype={'time':np.float64, 'type':np.int32, 'ord_id':np.int32, 
                   'size':np.int64, 'price':np.int64, 'dir':np.int32})
    
    df2=pd.read_csv(obfile, sep=',', header=None)
    
    df2_cols1 =['ask_prc_L1', 'ask_sz_L1', 'bid_prc_L1', 'bid_sz_L1', \
               'ask_prc_L2', 'ask_sz_L2', 'bid_prc_L2', 'bid_sz_L2', \
               'ask_prc_L3', 'ask_sz_L3', 'bid_prc_L3', 'bid_sz_L3', \
               'ask_prc_L4', 'ask_sz_L4', 'bid_prc_L4', 'bid_sz_L4', \
               'ask_prc_L5', 'ask_sz_L5', 'bid_prc_L5', 'bid_sz_L5']
    
    df2_cols2 =['ask_prc_L6', 'ask_sz_L6', 'bid_prc_L6', 'bid_sz_L6', \
               'ask_prc_L7', 'ask_sz_L7', 'bid_prc_L7', 'bid_sz_L7', \
               'ask_prc_L8', 'ask_sz_L8', 'bid_prc_L8', 'bid_sz_L8', \
               'ask_prc_L9', 'ask_sz_L9', 'bid_prc_L9', 'bid_sz_L9', \
               'ask_prc_L10', 'ask_sz_L10', 'bid_prc_L10', 'bid_sz_L10']
    
    df2.columns = df2_cols1 + df2_cols2
    
    df2.astype('int64')
    
    df = pd.concat([df1, df2], axis=1)
    '''
    darr= np.asarray(df, 
             dtype=[('time','f8'),('type','i4'),('ord_id','i4'), 
                    ('size','i8'),('price','i8'),('dir','i4'),
                    ('ask_prc_L1','f8'), ('ask_sz_L1','f8'),('bid_prc_L1','f8'),('bid_sz_L1','f8'),
                    ('ask_prc_L2','f8'), ('ask_sz_L2','f8'),('bid_prc_L2','f8'),('bid_sz_L2','f8'),
                    ('ask_prc_L3','f8'), ('ask_sz_L3','f8'),('bid_prc_L3','f8'),('bid_sz_L3','f8'),
                    ('ask_prc_L4','f8'), ('ask_sz_L4','f8'),('bid_prc_L4','f8'),('bid_sz_L4','f8'),
                    ('ask_prc_L5','f8'), ('ask_sz_L5','f8'),('bid_prc_L5','f8'),('bid_sz_L5','f8')])
    
    '''
    print ('Data Loaded:', df.shape, 'in', round(time()-t1,4), 'seconds')
    
    if format == 'pd':
        return df
    elif format == 'np':
        return df.as_matrix()
    else:
        print ("Warning: No format specified to LoadData()")

def spread_data(data):
    
    spread = data[:,0] - data[:,2]
    labels = np.diff(spread)
    labels[labels>0] = 1
    labels[labels<0] = -1
    data = data[:-1]
    #print ("Spread Classes 0, 1, -1: ", zeros, ones, negs)   
    #print ([labels])
    #labels = np.append(labels, [])    
    #print (data.shape, spread.shape, labels.shape)

    return data, labels



def remove_0_class (data, labels, dataonly=False):
    nz_ind= np.nonzero(labels)
    #print (z_ind)
    #print (labels[z_ind])    
    nz_labels = labels[nz_ind]
    nz_data = data[nz_ind]
   
    if dataonly==False:
        return nz_data, nz_labels
    else: 
        return nz_data

def Run_Experiment (exp_name, exp_ds, bal=None, algos='all', tune='n'):
     
    rdf = pd.DataFrame(np.empty((0,12)))
    rdf.columns = ['Experiment', 'Dataset', 'Classfr', 'Type', 
                       'Accuracy', 'Precision', 'Recall', 'F1_Weighted', 'F1_Macro',
                       'Acc0', 'Acc+1', 'Acc-1']#, 'F1_Micro']    
    
    for ds_name, X, y in exp_ds:
        print ("************\nExperiment:{} Dataset:{} Size:{}\n************".format(exp_name, ds_name, y.size))    
        
        if bal == None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        '''
        if bal=='smooth':
            smooth_data = trns.smooth_data(X, k_list=[20], min_list=[1], debug='n')
            #print (smooth_data)
            X = smooth_data[0][1]
            y = smooth_data[0][2]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            ds_name = ds_name + '-SMOO'
        '''
        
        if bal=='smote':
            smote = SMOTE(random_state=15)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)        
            X_train, y_train = smote.fit_sample(X_train, y_train)
            ds_name = ds_name + '-usos' 
            flog.write("Bal Labels O:{}, 1:{}, -1:{}".format(*trns.label_freq(y))) 
        
        #Run Classifiers
        results_train, results_test = clsf.RunClassifiers(X_train, X_test, y_train, y_test, experiment=exp_name, dataset=ds_name, algos=algos, tune=tune)
        
        ds_size = round((y.size) / 1000,0)
        
        fpref = fpath + '_' + exp_name + '_' + ds_name + '_sz' + repr(ds_size) + 'k'
        ftrain = fpref + '_TRAIN_.csv'
        ftest = fpref + '_TEST_.csv' 
        
        results_train.to_csv(ftrain)
        results_test.to_csv(ftest)
        
        rdf = pd.concat([rdf, results_train, results_test], ignore_index=True)
        
    rdf_train = rdf.loc[(rdf['Type'] == 'train')] 
    rdf_test = rdf.loc[(rdf['Type'] == 'test')]
    print (rdf_train, "\n")
    print (rdf_test, "\n")
    
    f1train = fpath + '_' + exp_name + '_TRAIN_.csv'
    f1test =  fpath + '_' + exp_name + '_TEST_.csv'
    rdf_train.to_csv(f1train)
    rdf_test.to_csv(f1test)
    
    #rdfback= pd.read_csv(f)
    #print (rdfback)
    
    return rdf


def main():
   
    
    stock = 'AMZN'
    lob_level = 10
    sample_size = 50000
    
    data_master = load_data (stock, lob_level, 'np')
    
    # Run for a smaller dataset ... put into load_data function  in future
    data_size = data_master[:,0].size
    sample_start = randint(0,data_size-sample_size)
    data_master = np.array(data_master[sample_start:sample_start+sample_size,:])
    
    flog.write ('Run for stock={}, lobs={}, datasize={}\n************\n\n'.format(stock, lob_level, data_master.shape))
    flog.flush()
    
    # Generate Labels based on price direction at next time, t+1
    t1 = time()
    X, y = trns.quote_data_labels(data_master)
    print ('Data Master:{}, Quote Data:{} Labels:{}'.format(data_master.shape, X.shape, y.shape))
    dsets = {'All': ('All', X,y)}
    
    # to quickly test some data
    t2 = time()
    X_test = X[:,4].reshape(y.size,1)
    dsets['Test'] = ('Test', X_test, y)
    
    # "random noise" dataset
    t3 = time()
    X_noise = np.random.rand(y.size,X[0].size)
    dsets['Noise'] = ('Noise',X_noise,y)
    print ("Generated Random Dataset in...", round(time()-t3,2))
    
    # Transformations: binary (+1/-1) class data (remove zero class)
    t4 = time()
    X_bin, y_bin = remove_0_class (X, y, dataonly=False)
    X_bin_noise = remove_0_class (X_noise, y, dataonly=True)
    dsets['Binary'] = ('Binary',X_bin,y_bin)
    dsets['BinaryNoise'] = ('BinaryNoise',X_bin_noise, y_bin)
    print ("Generated Binary Datasets...", round(time()-t4,2))
    
    smooth_ds=[]
    
    # Run smoothing window transformation
    t5 = time()
    
    '''
    smooth_data_list = trns.smooth_data(X, k_list=[5,10,20], min_list=[.5,1,1.5,2,2.5], debug='y')
    for i in smooth_data_list:
        dsets[i[0]] = (i[0], i[1], i[2])
        smooth_ds.append(dsets[i[0]]) 
    #flog.write ('------------------------------------ \n')
    #for i in smooth_ds:
    #    flog.write ('Smoothing {} \n Label Ratios {} \n'.format(i[0], itemfreq(i[2])))
    flog.write ('Smoothing Transforms run in: {} sec  \n'.format(round(time()-t5,2)))    
    flog.flush()
    print ("Generated Smooth Datasets...", round(time()-t5,2))
    '''
    
    # Get Smooth Data and Labelsfor further transforms
    smooth_data = trns.smooth_data(X, k_list=[20], min_list=[1], debug='n')
    X_smoo = smooth_data[0][1]
    y_smoo = smooth_data[0][2]
    dsets['All-smoo'] = ('All-smoo', X_smoo, y_smoo)
    print ("Labels O:{}, 1:{}, -1:{}".format(*trns.label_freq(y))) 
    flog.write("\nLabels O:{}, 1:{}, -1:{}\n".format(*trns.label_freq(y))) 
    print("Smoothed Labels O:{}, 1:{}, -1:{}".format(*trns.label_freq(y_smoo))) 
    flog.write("Smoothed Labels O:{}, 1:{}, -1:{}\n".format(*trns.label_freq(y))) 
            
    
    # Run ORD Transformations
    t6 = time()
    order_ds = []
    ord_data_list = trns.Order_Data(X_smoo)  # returns list of (dataset_name, dataset)
    for i in ord_data_list:
        dsets[i[0]] = (i[0], i[1], y_smoo)
        order_ds.append(dsets[i[0]])
    flog.write ('Order Transforms run in: {} sec  \n'.format(round(time()-t6,2)))    
    flog.flush()
    print ("Generated Order Datasets...", round(time()-t6,2))
    
    # Run LOB Transformations
    t7 = time()
    lob_ds = []
    lob_data_list = trns.LOB_Depths(X_smoo)
    for i in lob_data_list:
        dsets[i[0]] = (i[0], i[1], y_smoo)
        lob_ds.append(dsets[i[0]])
    flog.write ('LOB Transforms run in: {} sec  \n'.format(round(time()-t7,2)))    
    flog.flush()
    print ("Generated LOB Datasets...", round(time()-t7,2))
    
    # Run Imbalance Transformations
    t8 = time()
    imb_ds = []
    imb_data_list = trns.LOB_Imbalance(X_smoo)
    for i in imb_data_list:
        dsets[i[0]] = (i[0], i[1], y_smoo)
        imb_ds.append(dsets[i[0]])
    X_imb10 = dsets['IMB-10'][1]
    flog.write ('IMB Transforms run in: {} sec  \n'.format(round(time()-t8,2)))    
    flog.flush()
    print ("Generated Imbalance Datasets...", round(time()-t8,2))
    
    # Run Normlaization Transformations
    t9 = time()
    X_norm, X_imb10_norm = trns.Normalize_Data (X_smoo, X_imb10)
    print (X_norm.shape, X_imb10_norm.shape, y_smoo.shape)
    dsets['Norm-All'] = ('Norm-All',X_norm,y_smoo)
    dsets['Norm-Imb'] = ('Norm-Imb',X_imb10_norm,y_smoo)
    flog.write ('Normalization Transforms run in: {} sec \n'.format(round(time()-t9,2)))
    flog.flush()
    print ("Generated Normalization Datasets...", round(time()-t9,2))
    
    dsets['Lob-Rate'] = trns.LOB_Change_Rates(X_smoo, y_smoo, t_thresh=10)
    
    arrt_ds = []
    '''
    # Run volume rate transformations
    t10 = time()
    X_arrt_01,  X_arrt_ords_01, X_arrt_lobords_01, y_arrt_01 = trns.Order_Arrival_Rates(X_smoo, y_smoo, t_thresh=.1)
    X_arrt_1,  X_arrt_ords_1, X_arrt_lobords_1, y_arrt_1 = trns.Order_Arrival_Rates(X_smoo, y_smoo, t_thresh=1)
    X_arrt_10,  X_arrt_ords_10, X_arrt_lobords_10, y_arrt_10 = trns.Order_Arrival_Rates(X_smoo, y_smoo, t_thresh=10)
    #X_arrt_60,  X_arrt_ords_60, X_arrt_lobords_60, y_arrt_60 = trns.Order_Arrival_Rates(X, y, t_thresh=60)
   
    dsets['ArrRt-All-01'] = ('Arr-Rt-01',X_arrt_01,y_arrt_01)
    dsets['ArrRt-Ords-01'] = ('ArrRt-Ords-01',X_arrt_ords_01,y_arrt_01)
    dsets['ArrRt-LOBOrds-01'] = ('ArrRt-LOBOrds-01',X_arrt_lobords_01,y_arrt_01)
    dsets['ArrRt-All-1'] = ('Arr-Rt-1',X_arrt_1,y_arrt_1)
    dsets['ArrRt-Ords-1'] = ('ArrRt-Ords-1',X_arrt_ords_1,y_arrt_1)
    dsets['ArrRt-LOBOrds-1'] = ('ArrRt-LOBOrds-1',X_arrt_lobords_1,y_arrt_1)
    dsets['ArrRt-All-10'] = ('Arr-Rt-10',X_arrt_10,y_arrt_10)
    dsets['ArrRt-Ords-10'] = ('ArrRt-Ords-10',X_arrt_ords_10,y_arrt_10)
    dsets['ArrRt-LOBOrds-10'] = ('ArrRt-LOBOrds-10',X_arrt_lobords_10,y_arrt_10)
    #dsets['ArrRt-All-60'] = ('Arr-Rt-60',X_arrt_60,y_arrt_60)
    #dsets['ArrRt-Ords-60'] = ('ArrRt-Ords-60',X_arrt_ords_60,y_arrt_60)
    #dsets['ArrRt-LOBOrds-60'] = ('ArrRt-LOBOrds-60',X_arrt_lobords_60,y_arrt_60)
    
    arrt_ds = [dsets['ArrRt-All-01'],dsets['ArrRt-Ords-01'],dsets['ArrRt-LOBOrds-01'],
               dsets['ArrRt-All-1'],dsets['ArrRt-Ords-1'],dsets['ArrRt-LOBOrds-1'],
               dsets['ArrRt-All-10'],dsets['ArrRt-Ords-10'],dsets['ArrRt-LOBOrds-10']]
               #dsets['ArrRt-All-60'],dsets['ArrRt-Ords-60'],dsets['ArrRt-LOBOrds-60']]
               
    flog.write ('Arrival Rate Transforms run in: {} sec \n'.format(round(time()-t10,2)))
    flog.flush()
    print ("Generated Arrival Rate Datasets...", round(time()-t10,2))
    '''
    
    dsets['BigDataset'] = []
    '''
    #XBIG = np.concatenate ((X, XB5, X_norm), axis = 1)
    t11 = time()
    X_arrt = trns.Order_Arrival_Rates(X_smoo, y_smoo, t_thresh=1)
    X_big, y_big = trns.Combine_Tranformed_Data (X_smoo, X_imb10, X_arrt[0], y_smoo)
    dsets['BigDataset'] = ('BigDataset',X_big, y_big)
    flog.write ('Big Dataset Transform run in: {} sec \n'.format(round(time()-t11,2)))
    flog.flush()
    print ("Generated Big Dataset...", round(time()-t11,2))
    '''
    
    experiment = [('Initial', [dsets['Noise'], dsets['All']], None, 'all', 'n'),
     ('Binary', [dsets['Binary']], None, 'all', 'n'),
     ('OSUS', [dsets['All']], 'smote', 'all', 'n'),
     ('Smooth', smooth_ds, None, 'RF', 'n'), 
     ('Tune', [dsets['All']], None, 'RF', 'y'),
     ('Base', [dsets['All-smoo']], None, 'RF', 'n'),
     ('Orders', order_ds, None, 'RF', 'n'),
     ('LOBs', lob_ds, None, 'RF', 'n'),
     ('LOB Imbalances', imb_ds, None, 'RF', 'n'),
     ('Normalize', [dsets['Norm-All'], dsets['Norm-Imb']], None, 'RF', 'n'),
     ('Arrival Rates', arrt_ds, None, 'RF', 'n'),
     ('Lob Rates',[dsets['Lob-Rate']], None, 'RF', 'n'),
     ('Big Dataset',[dsets['BigDataset']], None, 'RF', 'n')]

    run_init = [experiment[0]]
    run_bin = [experiment[1]]
    run_osus = [experiment[2]]
    run_smth  = [experiment[3]]
    run_tune = [experiment[4]]
    run_base = [experiment[5]]
    run_ord = [experiment[6]]
    run_lob = [experiment[7]]
    run_imb = [experiment[8]]
    run_o2i = [experiment[6], experiment[7], experiment[8]]
    run_norm = [experiment[9]]
    run_arrt = [experiment[10]]
    run_lobrt =  [experiment[11]]
    run_big = [experiment[12]]
    run_after_tune = run_o2i+run_norm+run_arrt+run_big

    results_all = pd.DataFrame(np.empty((0,12)))
    results_all.columns = ['Experiment', 'Dataset', 'Classfr', 'Type', 
                      'Accuracy', 'Precision', 'Recall', 'F1_Weighted', 'F1_Macro',
                      'Acc0', 'Acc+1', 'Acc-1']
    
    for r in run_osus: 
        t_run = time()
        flog.write('-----------------------------------------------\n')
        flog.write('Run Experiment={} Bal={}, Algos={}, Tune={}\n'.format(r[0], r[2], r[3], r[4]))
        res = Run_Experiment(r[0], r[1], r[2], r[3], r[4])
        results_all = pd.concat([results_all, res], ignore_index=True)   
        flog.write('Completed in {} sec\n'.format(time()-t_run))
        flog.flush
        print ('---------------------------------')
        print('Experiment {} Completed in {} sec\n'.format(r[0], time()-t_run))
    #print (results_all)
    fres = fpath + '_AllResults.csv'
    results_all.to_csv (fres)

if __name__ == '__main__':
    main()

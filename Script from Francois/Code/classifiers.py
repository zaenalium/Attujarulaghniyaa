# -*- coding: utf-8 -*-
"""

@author: FQ
"""
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import os
cwd = os.path.abspath(os.path.curdir) 

from scipy.stats import itemfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import f1_score, make_scorer



np.random.seed(184)

def SetupClassifiers (algos = 'all'):
    
    #Parametersto tune 
    SVMGrad_parms = {'loss': ['hinge', 'log'], 
                     'alpha': [.0000001, .000001, .00001, .0001], 
                     'max_iter':[5,1000]}
    
    RF_parms = {'n_estimators':[100], 'max_features': ['sqrt', .5, 1], 
                'min_samples_leaf': [5,50]}
    
    #Introduction to Neural Networks for Java, Second Edition The Number of Hidden Layers
   
    NNet_parms = { 'hidden_layer_sizes': [(100,), (100,100), (50,), (200,)],
                  'alpha': [.00001, .0001, .001]}
    
    #https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    ADB_parms = {'learning_rate':[.05, .1, .2], 'n_estimators':[20,50,80]} 
    GDB_parms = {'learning_rate':[.05, .1, .2], 'n_estimators':[20,50,80]}
    
    Rand_parms = []
    GNB_parms = []
    GPC_parms = []
   

     #Classifiers List to iterate through'''
    clsfr = [('Rand', RandomClassifier(), Rand_parms),
             ('GNB', GaussianNB(), GNB_parms), 
             ('SVMGrad', SGDClassifier(alpha = .0000001, loss = 'hinge', max_iter=50), SVMGrad_parms),
             ('NNet', MLPClassifier(alpha = 0.0001, hidden_layer_sizes = (100,)), NNet_parms),
             ('ADB', AdaBoostClassifier(), ADB_parms),
             ('RF', RandomForestClassifier(max_features = 0.5, min_samples_leaf = 5, n_estimators = 100), RF_parms),
             ('GDB', GradientBoostingClassifier(), GDB_parms)]
             #('SVM', SVC(), SVM_parms),
             #('GPC', GaussianProcessClassifier(), GPC_parms)]

    if algos =='all':
        classifiers = clsfr
    elif algos == 'top3':
        classifiers = [clsfr[0],clsfr[4],clsfr[5],clsfr[6]]
    elif algos == 'top2':
        classifiers = [clsfr[0],clsfr[5],clsfr[6]]
    elif algos == 'RF':
        classifiers = [clsfr[0],clsfr[5]]
    elif algos == 'NN':
        classifiers = [clsfr[0],clsfr[3]]
    return classifiers

def RunClassifiers(data_train, data_test, labels_train, labels_test, experiment, dataset, algos='all', tune='n'):
    
    start_dttm = datetime.today().strftime('%y%m%d_%H%M')
    fname = 'RunClassifierLog_' + start_dttm + '.txt'
    f = open(fname, 'a')
    f.write ("Starting RunClassifiers() @ {} on train datasize:{}\n".format(start_dttm, data_train.shape))
   
    classifiers = SetupClassifiers(algos)
    f.write ("Completed Classifier Setup\n")
    
    num_res = 2*len(classifiers)
    res = 0
    results = pd.DataFrame(np.empty((num_res,12)))
    results.columns = ['Experiment', 'Dataset', 'Classfr', 'Type', 
                       'Accuracy', 'Precision', 'Recall', 'F1_Weighted', 'F1_Macro', 'Acc0', 'Acc+1', 'Acc-1']
    
    # run classifiers
    for c, func, tune_parms in classifiers:
        
        t0 = time()
        f.write("--------------------------\nExperiment:{} Dataset:{} Tuning?:{} Algo:{} ...\n".format(experiment, dataset, tune, c))
        
        model = func 
        if tune == 'y' and not (c == 'Rand' or c=='GNB'):
            print ('{} :: {}, {}, Tuning:{}'.format(time(), experiment, dataset, c))
            model = GridSearchCV(func, tune_parms, scoring='f1_macro')
        else:
            print ('{} :: {}, {}, Fitting:{}'.format(time(), experiment, dataset, c))
            model = func
            
        model.fit (data_train,labels_train)
        if tune == 'y' and not (c == 'Rand' or c=='GNB'):  print(model.best_params_)
        t1 = time()
        f.write("Fitted in {} sec\n".format(round(t1-t0, 4)))
        
        
        '''
        if c == 'RF':
            f.write("Important Feautures{}".format(model.feature_importances_))
        '''
        
        pred_train = model.predict(data_train)
        pred_test = model.predict(data_test)
        
        #print (pred_test, labels_test)
        
        t2 = time()
        f.write("Predictions in {} sec\n".format(round(t2-t1, 4)))
        
        acc_train = ((pred_train == labels_train).mean()) 
        acc_test = ((pred_test == labels_test).mean())
        P_train, R_train, F1_train_w, sup_train = precision_recall_fscore_support(labels_train, pred_train, pos_label=None, average='weighted', warn_for=('recall'))
        P_test, R_test, F1_test_w, sup_test = precision_recall_fscore_support(labels_test, pred_test, pos_label=None, average='weighted', warn_for=('recall'))
        
        F1_train_mac = f1_score (labels_train, pred_train, pos_label=None, average='macro')
        F1_test_mac = f1_score (labels_test, pred_test, pos_label=None, average='macro')
        
        F1_train_mic = f1_score (labels_train, pred_train, pos_label=None, average='micro')
        F1_test_mic = f1_score (labels_test, pred_test, pos_label=None, average='micro')
        
        CM_train = confusion_matrix(labels_train, pred_train, labels=[-1,1,0])
        CM_test = confusion_matrix(labels_test, pred_test, labels=[-1,1,0])
        
        #print ("Label Frequency Train:", itemfreq(labels_train))
        #print ("Label Frequency Test:", itemfreq(labels_test))
        #print ("Confusion Matrix Train", CM_train)
        #print ("Confusion Matrix Test", CM_test)
        
        acc_train_m1 = CM_train[0,0] / np.sum(CM_train[0,:])
        acc_train_p1 = CM_train[1,1] / np.sum(CM_train[1,:])
        acc_test_m1 = CM_test[0,0] / np.sum(CM_test[0,:])
        acc_test_p1 = CM_test[1,1] / np.sum(CM_test[1,:])
        
        #don't calculate for binary labels case
        if CM_train[0].size == 3: 
            acc_train_0 = CM_train[2,2] / np.sum(CM_train[2,:])
            acc_test_0 = CM_test[2,2] / np.sum(CM_train[2,:])
        else:
            acc_train_0 = 'n/a'
            acc_test_0 = 'n/a'
            
        results.iloc[res] = [experiment, dataset, c, 'train', acc_train, P_train, R_train, F1_train_w, F1_train_mac, acc_train_0, acc_train_p1, acc_train_m1]#, F1_train_mic] 
        results.iloc[res+1] = [experiment, dataset, c, 'test', acc_test, P_test, R_test, F1_test_w, F1_test_mac, acc_test_0, acc_test_p1, acc_test_m1]#, F1_test_mic] 
        res = res+2
        
        t3 = time()
        f.write("{} completed in {} sec\n".format(c, round(t3-t0, 4)))
        
    #results.to_csv('results.txt', sep=' ')  
    #print (results.loc[(results['Dataset'] == 'train')], "\n")
    #print (results.loc[(results['Dataset'] == 'test')], "\n")
    f.close()
    
    results_train = results.loc[(results['Type'] == 'train')]
    results_test = results.loc[(results['Type'] == 'test')]
    
    return results_train, results_test


def TuneClassifiers(data_train, data_test, labels_train, labels_test):
    
    '''Tuning Parameters'''
   
    
    SVMGrad_parms = {'loss': ['hinge', 'log'], 
                     'alpha': [.0000001, .000001, .00001, .0001], 
                     'max_iter':[1000,2000]}
    
    RF_parms = {'n_estimators':[10], 'max_features': ['sqrt', 'log2'], 
                'min_samples_leaf': [1,5,50]}
    
    #Introduction to Neural Networks for Java, Second Edition The Number of Hidden Layers
    n_x = data_train[0,:].size
    n_x1 = 4        # lower range for neurons in hidden layer
    n_x3 = n_x      # upper ramge for neurons in hidden layer
    n_x2 = int((n_x3 + n_x1)/2)
    NNet_parms = { 'hidden_layer_sizes': [(100,), (n_x1,), (n_x2,), (n_x3,)],
                  'alpha': [.00001, .0001, .001]}
    GNB_parms = []
    Rand_parms = []
    SVM_parms=[]
    
    '''Classifiers List to iterate through'''
    classifiers = [('GNB', GaussianNB(), GNB_parms), 
                   ('SVMGrad', SGDClassifier(), SVMGrad_parms),
                   ('RF', RandomForestClassifier(), RF_parms),
                   ('NNet', MLPClassifier(), NNet_parms),
                   ('Rand', RandomClassifier(), Rand_parms)]
                   #('SVM', SVC(), SVM_parms),
                   #('GPC', GaussianProcessClassifier())]

    num_res = 2*len(classifiers)
    res = 0  
    tune_results = pd.DataFrame(np.empty((num_res,4)))
    tune_results.columns = ['Classifier', 'Dataset', 'Type', 'Accuracy']    
    
    # run the base (non-tuned) models
    
    
    for c, func, parameters in classifiers:
        model = func 
        model.fit (data_train,labels_train)
        pred_train = model.predict(data_train)
        pred_test = model.predict(data_test)
        
        acc_train = ((pred_train == labels_train).mean()) 
        acc_test = ((pred_test == labels_test).mean())
        P_train, R_train, F1_train, sup_train = 0, 0, 0,0  #precision_recall_fscore_support(labels_train, pred_train, average='macro')
        P_test, R_test, F1_test, sup_test = 0, 0, 0, 0 #precision_recall_fscore_support(labels_test, pred_test, average='macro')

        tune_results.loc[res] = [c, 'train', 'base', acc_train]
        tune_results.loc[res+1] = [c, 'test', 'base', acc_test]
        res = res+2
    
    # run model tuning
    
    for c, func, parameters in classifiers:
        
        t0 = time()
        print ("Tuning", c, "...")
        
        if not (c == 'Rand' or c=='GNB'):
            tun_model = GridSearchCV(func, parameters) 
        else:
            tun_model = func
        
        tun_model.fit (data_train,labels_train)
        
        t1 = time()
        print ("Tuned in:", t1-t0)
        
        if not (c == 'Rand' or c=='GNB'):  print(tun_model.best_params_)
        pred_train = tun_model.predict(data_train)
        pred_test = tun_model.predict(data_test)
        
        acc_train = ((pred_train == labels_train).mean()) 
        acc_test = ((pred_test == labels_test).mean())
        #P_train, R_train, F1_train, sup_train = 0, 0, 0,0  #precision_recall_fscore_support(labels_train, pred_train, average='macro')
        #P_test, R_test, F1_test, sup_test = 0, 0, 0, 0 #precision_recall_fscore_support(labels_test, pred_test, average='macro')

        tune_results.loc[res] = [c, 'train', 'tuned', acc_train]
        tune_results.loc[res+1] = [c, 'test', 'tuned', acc_test]
        res = res+2 
    
    #tune_results.to_csv('tuning_results.txt', sep=' ')  
    
    results_train = results.loc[(results['Type'] == 'train')]
    results_train = results.loc[(results['Type'] == 'test')]
    
    return tune_results
        
class RandomClassifier(object):
    def __init__(self):
        self.ones = 0
        self.zeros = 0
        self.negs = 0
        
    def fit(self, data, labels):
        num_ones = labels[labels==1].size
        num_negs = labels[labels==-1].size
        num_zeros = labels[labels==0].size
        total = labels.size
        self.ones = num_ones / total
        self.negs = num_negs / total
        self.zeros = 1 - self.ones - self.negs
        if self.zeros < 0: self.zeros = 0
        
    def predict(self, data):
        num_preds = data[:,0].size
        rand_labels = np.random.choice ([0,1,-1], size=(num_preds), p=[self.zeros, self.ones, self.negs] )
        return rand_labels
    

    
    



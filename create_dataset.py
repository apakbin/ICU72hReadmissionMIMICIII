# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:04:50 2018

@author: a.pakbin
"""
import numpy as np
from copy import copy
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random as rnd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost.sklearn import XGBClassifier
import sys
import os
import matplotlib.pyplot as plt

import re

def data_reader(data_address, file_name, non_attribute_column_names=None,label_column_name=None):
    
    data=pd.read_csv(data_address+'/'+file_name)
    if non_attribute_column_names:
        columns_to_drop=list(set(non_attribute_column_names)-set([label_column_name]))
        data=data.drop(columns_to_drop, axis=1)
    return data

def matrix_partitioner(df, proportion, label=None):
    
    number_of_ones=int(round(proportion*len(df)))
    ones=np.ones(number_of_ones)
    zeros=np.zeros(len(df)-number_of_ones)
    ones_and_zeros=np.append(ones,zeros)
    permuted=np.random.permutation(ones_and_zeros)
    boolean_permuted=permuted>0
    
    if label:
        return [df[boolean_permuted].reset_index(),df[~boolean_permuted].reset_index(),label[boolean_permuted],label[~boolean_permuted]]
    else:
        return [df[boolean_permuted].reset_index(),df[~boolean_permuted].reset_index()]


def dataframe_partitioner(df, output_label, proportion):
    y=df[output_label].values
    X=df.drop([output_label], axis=1)
    
    return matrix_partitioner(X,label=y,proportion=proportion)

def one_hot_detacher(X, categorical_column_names):
    one_hot_column_names=list()
    for categorical_column in categorical_column_names:
            for column_name in X.columns:
                if column_name.startswith(categorical_column):
                    one_hot_column_names.append(column_name)
    one_hot=X[one_hot_column_names]
    X.drop(one_hot_column_names, axis=1, inplace=True)
    return [X, one_hot]

def one_hot_attacher(X, one_hot):
    return X.join(one_hot)

def normalize(X, data_type, categorical_column_names, training_mean=None, training_std=None):
    [X, one_hot]=one_hot_detacher(X, categorical_column_names)
    if data_type=='train_set':
        mean=np.mean(X,axis=0)
        std=np.var(X, axis=0)   	
    elif data_type=='test_set':
        mean=training_mean
        std=training_std
        
    aux_std=copy(std)
    aux_std[aux_std==0]=1
    normalized=(X-mean)/aux_std
    
    complete_normalized=one_hot_attacher(normalized, one_hot)
    
    if data_type=='train_set':
        return [complete_normalized, mean, std]  	
    elif data_type=='test_set':
        return complete_normalized

def train_test_normalizer(X_train, X_test, categorical_column_names):

    [X_TRAIN_NORMALIZED, X_TRAIN_MEAN, X_TRAIN_STD]=normalize(X=X_train, data_type='train_set', categorical_column_names=categorical_column_names)
    X_TEST_NORMALIZED=normalize(X=X_test, data_type='test_set', categorical_column_names=categorical_column_names, training_mean=X_TRAIN_MEAN, training_std=X_TRAIN_STD)

    return [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]
'''
def categorical_imputer(X, categorical_column_names, to_be_imputed_one=None, related_columns=None):
    if to_be_imputed_one is None:
        related_columns=list()
        for categorical_column in categorical_column_names:
            new_categorical_column=list()
            for column_name in X.columns:
                if column_name.startswith(categorical_column):
                    new_categorical_column.append(column_name)
            related_columns.append(new_categorical_column)
            
        to_be_imputed_one=list()
        for categorical_variable in related_columns:
            maximum=-1
            value_with_max=None
            for value in categorical_variable:
                mean=X[value].mean()
                if (mean > maximum):
                    maximum=mean
                    value_with_max=value
            to_be_imputed_one.append(value_with_max)
            
        for value in to_be_imputed_one:
            X[value].fillna(1,inplace=True)
        
        for categorical_variable in related_columns:
            for value in categorical_variable:
                X[value].fillna(0,inplace=True)
        return [X,to_be_imputed_one, related_columns]
    else:
        for value in to_be_imputed_one:
            X[value].fillna(1,inplace=True)
        
        for categorical_variable in related_columns:
            for value in categorical_variable:
                X[value].fillna(0,inplace=True)
        return X
'''
## impute categorical value


def possible_values_finder(data, categorical_column_names):
    column_dict = dict()
    for categorical_column_name in categorical_column_names:
        unique_vals = list(set([str(x) for x in data[categorical_column_name].unique()])-set(['nan','NaN','NAN','null']))
        column_dict[categorical_column_name]=unique_vals
    return column_dict
        

def one_hot_encoder(X, categorical_column_names, possible_values):
    for categorical_column_name in categorical_column_names:
        possible_values_ = possible_values[categorical_column_name]
        new_vals = [categorical_column_name + '_' + str(s) for s in possible_values_]
        dummies = pd.get_dummies(X[categorical_column_name], prefix=categorical_column_name)
        dummies = dummies.T.reindex(new_vals).T.fillna(0)
        X = X.drop([categorical_column_name], axis=1)
        X = X.join(dummies)
    return X

def train_test_one_hot_encoder(X_train, X_test, categorical_column_names, possible_values):
    X_TRAIN=one_hot_encoder(X_train, categorical_column_names, possible_values)
    X_TEST=one_hot_encoder(X_test, categorical_column_names, possible_values)
    return [X_TRAIN, X_TEST]

def categorical_distribution_finder(X, categorical_column_names):
    NAMES=list()
    DISTS=list()
    for categorical_column_name in categorical_column_names:
        names=list()
        nom_of_all=0
        quantity=list()
        grouped= X.groupby([categorical_column_name])
        for category, group in grouped:
            names.append(category)
            quantity.append(len(group))
            nom_of_all=nom_of_all+len(group)
        distribution = [float(x) / nom_of_all for x in quantity]

        NAMES.append(names)
        DISTS.append(distribution)
    return(NAMES, DISTS)
        
def categorical_imputer(X, categorical_column_names, data_type='train', names=None, distributions=None):
    if data_type=='train':
        [names, distributions]=categorical_distribution_finder(X, categorical_column_names)
    
    for idx, categorical_column_name in enumerate(categorical_column_names):
        for i in range(0, len(X)):
            if pd.isnull(X[categorical_column_name].iloc[i]):
                X[categorical_column_name].iloc[i]=np.random.choice(names[idx], p=distributions[idx])
    
    if data_type=='train':
        return [X, names, distributions]
    elif data_type=='test':
        return X

def numerical_imputer(X, training_mean=None):
    if training_mean is None:
        training_mean=X.mean()
        imputed=X.fillna(training_mean)
        return [imputed, training_mean]
    else:
        imputed=X.fillna(training_mean)
        return imputed
    
def train_test_imputer(X_train, X_test, categorical_column_names):
    
    [X_TRAIN_CAT_IMPUTED, NAMES, DISTS]=categorical_imputer(X_train, categorical_column_names)
    X_TEST_CAT_IMPUTED=categorical_imputer(X_test, categorical_column_names, 'test', NAMES, DISTS)
    
    [X_TRAIN_IMPUTED, X_TRAIN_MEAN]=numerical_imputer(X_TRAIN_CAT_IMPUTED)
    X_TEST_IMPUTED=numerical_imputer(X_TEST_CAT_IMPUTED, X_TRAIN_MEAN)
    
    return [X_TRAIN_IMPUTED, X_TEST_IMPUTED]

def auc_calculator(model, X, y, num_of_folds):
    auc_list=list()
    skf=StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=rnd.randint(1,1e6))
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
   
        model.fit(X_train, y_train)
        predictions=model.predict_proba(X_test)[:,1]
        auc=roc_auc_score(y_true=y_test, y_score=predictions)
        auc_list.append(auc)

    return sum(auc_list)/len(auc_list)    

def grid_search(model_type, X, y, num_of_folds, verbose, first_dim, second_dim=None, third_dim=None, return_auc_values=False):
    
    best_auc=0
    best_auc_setting=None
    
    if model_type=='XGB':        
        auc_matrix=np.zeros((len(first_dim),len(second_dim),len(third_dim)))
        for max_depth_index, max_depth in enumerate(first_dim):
            for n_estimator_index, n_estimator in enumerate(second_dim):
                for learning_rate_index, learning_rate in enumerate(third_dim):
                    
                    model=XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimator), learning_rate=learning_rate)
                    auc=auc_calculator(model, X, y, num_of_folds)
                    auc_matrix[max_depth_index, n_estimator_index, learning_rate_index]=auc
                    if auc>best_auc:
                        best_auc=auc
                        best_auc_setting=[max_depth,n_estimator,learning_rate]
                    if verbose==True:
                        sys.stdout.write('\rGRID SEARCHING XGB: progress: {0:.3f} % ...'.format(
                                (max_depth_index*(len(second_dim)*len(third_dim))+
                                 n_estimator_index*(len(third_dim))+
                                 learning_rate_index
                                 +1)/(len(first_dim)*len(second_dim)*len(third_dim))*100))
    if model_type=='LR+LASSO':
        auc_matrix=np.zeros(len(first_dim))
        for index, regularization_strength in enumerate(first_dim):
            model=LogisticRegression(penalty='l1', C=regularization_strength)
            auc=auc_calculator(model, X, y, num_of_folds)
            auc_matrix[index]=auc
            if auc>best_auc:
                best_auc=auc
                best_auc_setting=regularization_strength
            if verbose==True:        
                sys.stdout.write('\rGRID SEARCHING LR: progress: {0:.3f} % ...'.format((index+1)/(len(first_dim))*100))
    
    if model_type=='SGD':
        auc_matrix=np.zeros(len(first_dim))
        for index, alpha in enumerate(first_dim):
            model=SGDClassifier(penalty='l1', loss='log', shuffle=True, alpha=alpha, class_weight='balanced', n_jobs=-1)
            auc=auc_calculator(model, X, y, num_of_folds)
            auc_matrix[index]=auc
            if auc>best_auc:
                best_auc=auc
                best_auc_setting=alpha
            if verbose==True:        
                sys.stdout.write('\rGRID SEARCHING SGD: progress: {0:.3f} % ...'.format((index+1)/(len(first_dim))*100))
    
    if return_auc_values:
        return [best_auc_setting,auc_matrix]
    else:
        return best_auc_setting
    
def auc_for_different_feature_set_sizes(X, y, ranked_features, feature_set_sizes):
    num_of_folds=3
    X_sorted=X[ranked_features]
    auc_list=list()
    for idx, number_of_features in enumerate(feature_set_sizes):
        trimmed_X=X_sorted.iloc[:,0:number_of_features]
        [XXX,auc_matrix]=grid_search('XGB', trimmed_X, y, num_of_folds, False, np.linspace(start=1, stop=6, num=6), [100,200,500,1000], [0.1])
        
        auc_list.append(np.amax(auc_matrix))
        sys.stdout.write('\rCALCULATING AUROC CURVE FOR DIFFERENT FEATURE SIZES: progress: {0:.3f} % ...'.format((idx+1)/(len(feature_set_sizes))*100))
        
    return [feature_set_sizes, auc_list]

def auc_for_different_feature_sets(X, y, feature_sets):
    num_of_folds=3
    XGB_AUC=list()
    LR_AUC=list()
    for idx, feature_set in enumerate(feature_sets):
        
        trimmed_X=X[feature_set]
        '''
        [XXX,auc_matrix]=grid_search('XGB', trimmed_X, y, num_of_folds, True, np.linspace(start=1, stop=6, num=6), [100,200,500,1000], [0.1])
        XGB_AUC.append(np.amax(auc_matrix))
        '''
        #"""
        [XXX,auc_matrix]=grid_search('LR+LASSO', trimmed_X, y, num_of_folds, True, np.logspace(start=-5, stop=5, num=11))
        LR_AUC.append(np.amax(auc_matrix))
        #"""
        
        sys.stdout.write('\rCALCULATING AUROC CURVE FOR DIFFERENT FEATURE SETS: progress: {0:.3f} % ...'.format((idx+1)/(len(feature_sets))*100))
        
    return [XGB_AUC, LR_AUC]
'''   
def likelihood_ratio_test(features_alternate, labels):#, lr_model):
    
    lr_model=SGDClassifier(loss="log", penalty="l1", max_iter=max(20, int(np.ceil(10**6 / len(features_alternate)))))
    
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)
    

    null_prob = sum(labels) / float(labels.shape[0]) * \
                np.ones(labels.shape)
    df = features_alternate.shape[1]
    
    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    pvalue = chi2.sf(G, df)

    return pvalue

def p_values(X,y):
    pvalues=list()
    for i in range(0,len(X.columns)):
        column = X.iloc[:, i].values.reshape(-1, 1)
        pvalues.append(likelihood_ratio_test(column,y))
    
    features_and_pvalues=pd.DataFrame(
    {'feature': list(X.columns),
     'p value': pvalues
    })
    return features_and_pvalues
'''
def vectors_to_csv(address, file_name, vector_one, label_one, vector_two=None, label_two=None,vector_three=None, label_three=None):
    if vector_two is None:
        df=pd.DataFrame(data={label_one:vector_one})
    elif vector_three is None:
        df=pd.DataFrame(data={label_one:vector_one, label_two:vector_two})
    else:
        df=pd.DataFrame(data={label_one:vector_one, label_two:vector_two, label_three:vector_three})
    df.to_csv(address+'/'+file_name+'.csv')
    
def create_subfolder_if_not_existing(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def save_roc_curve(data_address, TPR, FPR, auc):
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(FPR, TPR, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
#    plt.show()
    plt.savefig(data_address)
    plt.close()

def combine_PREDICTIONs_and_save(current_folder, PREDICTIONs):
    PREDICTIONS=pd.concat(PREDICTIONs, axis=1)
    PREDICTIONS.to_csv(current_folder+'/'+'predictions.csv')

def feature_importance_updator(accumulative_feature_importance, new_importance):
    if accumulative_feature_importance is None:
        return new_importance
    else:
        return accumulative_feature_importance+new_importance
    
        
def feature_importance_saver(address, col_names, accumulative_feature_importance, num_of_folds):
    
    mean_feature_importances=accumulative_feature_importance/num_of_folds
    
    DF=pd.DataFrame(data={'col_name': col_names, 'importance': mean_feature_importances})
    DF.to_csv(address+'/'+'feature_importances.csv')
    DF=DF.sort_values(by='importance', ascending=False).reset_index(drop=True)
    DF.to_csv(address+'/'+'feature_importances_sorted.csv')

def first_matching_ICD9_finder(code, convertor_dict):
    ones=range(0,10)
    for one in ones:
        try:
            Matching_ICD9s_name=convertor_dict[10*code+one]
            return Matching_ICD9s_name
        except:
            continue
    return 'UNKNOWN'


def convert_ICD9_codes(features_list):
    ICD9Codes=pd.read_csv('data/DATA_ALL/D_ICD_PROCEDURES.csv.gz')

    
    convertor_dict=dict(zip(ICD9Codes['ICD9_CODE'],ICD9Codes['LONG_TITLE']))
    feature_names = ['ICD9_'+str(feature[5:])+'_'+ first_matching_ICD9_finder(int(feature[5:]), convertor_dict)
                     if feature.startswith('ICD9_')
                     else feature
                     for feature in features_list]
    return feature_names
    '''
    for feature in features_list:
        print (feature.isnone())
    #
    '''
def convert_items_n_labitems(features_list):
    RE_INT = re.compile(r'^[-+]?([1-9]\d*|0)$')
    df_D_ITEMS = pd.read_csv('data/DATA_ALL/D_ITEMS.csv.gz')
    df_D_LABITEMS = pd.read_csv('data/DATA_ALL/D_LABITEMS.csv.gz')
    df_items = pd.concat([df_D_ITEMS[['ITEMID','LABEL']], df_D_LABITEMS[['ITEMID','LABEL']]]).set_index('ITEMID')

    feature_names = [df_items.loc[int(feature.split('_')[0])].LABEL+' ('+feature.split('_')[1] + ')'
                     if RE_INT.match(feature.split('_')[0])
                     else feature for feature in features_list ]
    return feature_names
    
def convert_numbers_to_names(features_list):
    return convert_ICD9_codes(convert_items_n_labitems(features_list))
    
def categorical_finder(data, potential_categorical_suffix):
    col_names=list()
    nom_of_unique_vals=list()
    for col_name in data.columns:
        if col_name.endswith(potential_categorical_suffix):
            col_names.append(col_name)
            nom_of_unique_vals.append(len(data[col_name].unique()))
    col_names_unique_vals=pd.DataFrame(data={'colname': col_names, 'nom_of_unique_vals': nom_of_unique_vals})
    print (col_names_unique_vals.sort_values(by=['nom_of_unique_vals']))
''' 
def delete_erroneous_variables(data, erroneous_variables_codes):
    erroneous_col_names=list()
    for col_name in data.columns:
        for erroneous_variable_code in erroneous_variables_codes:
            if col_name.startswith(erroneous_variable_code):
                erroneous_col_names.append(col_name)
    return (data.drop(erroneous_col_names,axis=1))
'''    
'''
def ICD9_categorizer(X):
    ICD9_COLUMN_NAMES=[col for col in X.columns if str(col).startswith('ICD9_')]
    ICD9_categorized=pd.DataFrame(index=range(0,len(X)), columns=['ICD9_001~139','ICD9_140~239','ICD9_240~279','ICD9_280~289','ICD9_290~319','ICD9_320~389','ICD9_390~459','ICD9_460~519','ICD9_520~579','ICD9_580~629','ICD9_630~679','ICD9_680~709','ICD9_710~739','ICD9_740~759','ICD9_760~779','ICD9_780~799','ICD9_800~999']).fillna(0)

    for ICD9_column_name in ICD9_COLUMN_NAMES:
        index=int(int(ICD9_column_name[5:])/10)
        
        FITTING_CATEGORY=None
        for categorized_col_name in ICD9_categorized.columns:
            low_idx=categorized_col_name.find('_')
            lower_bound=int(categorized_col_name[low_idx+1:low_idx+4])
            high_idx=categorized_col_name.find('~')
            higher_bound=int(categorized_col_name[high_idx+1:high_idx+4])
            
            if lower_bound<=index and index<=higher_bound:
                FITTING_CATEGORY=categorized_col_name
                break
        
        ICD9_categorized[FITTING_CATEGORY]=ICD9_categorized[FITTING_CATEGORY]+X[ICD9_column_name]
        
    X=X.drop(ICD9_COLUMN_NAMES, axis=1)
    X=X.join(ICD9_categorized)
    return X
'''

'''
def ICD9_categorizer(X):
    ICD9_COLUMN_NAMES=[col for col in X.columns if str(col).startswith('ICD9_')]
    ICD9_categorized=pd.DataFrame(index=range(0,len(X)), columns=['ICD9_001~009','ICD9_010~018','ICD9_020~027','ICD9_030~041','ICD9_042~044','ICD9_045~049','ICD9_050~059','ICD9_060~066','ICD9_070~079','ICD9_080~088','ICD9_090~099','ICD9_100~104','ICD9_110~118','ICD9_120~129','ICD9_130~136','ICD9_137~139','ICD9_140~239','ICD9_240~279','ICD9_280~289','ICD9_290~319','ICD9_320~389','ICD9_390~392','ICD9_393~398','ICD9_401~405','ICD9_410~414','ICD9_415~417','ICD9_420~429','ICD9_430~438','ICD9_440~449','ICD9_451~459','ICD9_460~466','ICD9_470~478','ICD9_480~488','ICD9_490~496','ICD9_500~508','ICD9_510~519','ICD9_520~579','ICD9_580~629','ICD9_630~679','ICD9_680~709','ICD9_710~739','ICD9_740~759','ICD9_760~779','ICD9_780~799','ICD9_800~999']).fillna(0)

    for ICD9_column_name in ICD9_COLUMN_NAMES:
        index=int(int(ICD9_column_name[5:])/10)        
        FITTING_CATEGORY=None
        
        for categorized_col_name in ICD9_categorized.columns:
            low_idx=categorized_col_name.find('_')
            lower_bound=int(categorized_col_name[low_idx+1:low_idx+4])
            high_idx=categorized_col_name.find('~')
            higher_bound=int(categorized_col_name[high_idx+1:high_idx+4])
            
            if lower_bound<=index and index<=higher_bound:
                FITTING_CATEGORY=categorized_col_name
                break
        
        if FITTING_CATEGORY==None:
            continue
        
        ICD9_categorized[FITTING_CATEGORY]=ICD9_categorized[FITTING_CATEGORY]+X[ICD9_column_name]
        
    X=X.drop(ICD9_COLUMN_NAMES, axis=1)
    X=X.join(ICD9_categorized)
    return X
'''

def ICD9_categorizer(X):
    ICD9_COLUMN_NAMES=[col for col in X.columns if str(col).startswith('ICD9_')]
    ICD9_categorized=pd.DataFrame(index=range(0,len(X)), columns=['ICD9_'+str(x) for x in range(0,1000)]).fillna(0)

    for ICD9_column_name in ICD9_COLUMN_NAMES:
        index=int(int(ICD9_column_name[5:])/10)
        FITTING_CATEGORY='ICD9_'+str(index)
        ICD9_categorized[FITTING_CATEGORY]=ICD9_categorized[FITTING_CATEGORY]+X[ICD9_column_name]
        
    X=X.drop(ICD9_COLUMN_NAMES, axis=1)
    X=X.join(ICD9_categorized)
    return X
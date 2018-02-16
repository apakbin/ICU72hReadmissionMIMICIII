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

def grid_search(X, y, num_of_folds, verbose, first_dim, second_dim=None, third_dim=None, return_auc_values=False):
    
    best_auc=0
    best_auc_setting=None
          
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
                    sys.stdout.write('\r   GRID SEARCHING XGB: progress: {0:.3f} % ...'.format(
                            (max_depth_index*(len(second_dim)*len(third_dim))+
                             n_estimator_index*(len(third_dim))+
                             learning_rate_index
                             +1)/(len(first_dim)*len(second_dim)*len(third_dim))*100))

    print ('\n')
    if return_auc_values:
        return [best_auc_setting,auc_matrix]
    else:
        return best_auc_setting
    
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

def feature_importance_updator(accumulative_feature_importance, new_importance):
    if accumulative_feature_importance is None:
        return new_importance
    else:
        return accumulative_feature_importance+new_importance
     
def feature_importance_saver(address, col_names, accumulative_feature_importance, num_of_folds):
    
    mean_feature_importances=accumulative_feature_importance/num_of_folds
    
    DF=pd.DataFrame(data={'FEATURE': col_names, 'IMPORTANCE': mean_feature_importances})
    DF.to_csv(address+'/'+'feature_importances.csv')
    DF=DF.sort_values(by='IMPORTANCE', ascending=False).reset_index(drop=True)
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

def convert_ICD9_codes(features_list, conversion_tables_address):
    ICD9Codes=pd.read_csv(conversion_tables_address+'/'+'D_ICD_PROCEDURES.csv.gz')

    convertor_dict=dict(zip(ICD9Codes['ICD9_CODE'],ICD9Codes['LONG_TITLE']))
    feature_names = ['ICD9_'+str(feature[5:])+'_'+ first_matching_ICD9_finder(int(feature[5:]), convertor_dict)
                     if feature.startswith('ICD9_')
                     else feature
                     for feature in features_list]
    return feature_names

def convert_items_n_labitems(features_list, conversion_tables_address):
    RE_INT = re.compile(r'^[-+]?([1-9]\d*|0)$')
    df_D_ITEMS = pd.read_csv(conversion_tables_address+'/'+'D_ITEMS.csv.gz')
    df_D_LABITEMS = pd.read_csv(conversion_tables_address+'/'+'D_LABITEMS.csv.gz')
    df_items = pd.concat([df_D_ITEMS[['ITEMID','LABEL']], df_D_LABITEMS[['ITEMID','LABEL']]]).set_index('ITEMID')

    feature_names = [df_items.loc[int(feature.split('_')[0])].LABEL+' ('+feature.split('_')[1] + ')'
                     if RE_INT.match(feature.split('_')[0])
                     else feature for feature in features_list ]
    return feature_names
    
def convert_numbers_to_names(features_list, conversion_tables_address):
    return convert_ICD9_codes(convert_items_n_labitems(features_list, conversion_tables_address), conversion_tables_address)

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

def save_fold_data(writing_dir, fold_number, icustay_id_train, X_TRAIN_NORMALIZED, y_train, icustay_id_test, X_TEST_NORMALIZED, y_test, convert_names, conversion_tables_address=None):

    ICUSTAY_ID_TRAIN=pd.DataFrame(data={'ICUSTAY_ID': icustay_id_train})
    Y_TRAIN=pd.DataFrame(data={'LABEL': y_train})
    X_TRAIN_NORMALIZED=X_TRAIN_NORMALIZED.reset_index().drop(['index'],axis=1)
    TRAINING=pd.concat([ICUSTAY_ID_TRAIN, X_TRAIN_NORMALIZED, Y_TRAIN], axis=1)
    
    ICUSTAY_ID_TEST=pd.DataFrame(data={'ICUSTAY_ID': icustay_id_test})
    Y_TEST=pd.DataFrame(data={'LABEL': y_test})
    X_TEST_NORMALIZED=X_TEST_NORMALIZED.reset_index().drop(['index'],axis=1)
    TESTING=pd.concat([ICUSTAY_ID_TEST, X_TEST_NORMALIZED, Y_TEST], axis=1)
    
    if convert_names:
        changed_col_names=convert_numbers_to_names(TRAINING.columns, conversion_tables_address)
        TRAINING.columns=changed_col_names
        TESTING.columns =changed_col_names
    
    TRAINING.to_csv(writing_dir+'/'+'fold_'+str(fold_number)+'_'+'training_data.csv',index=False)
    TESTING.to_csv (writing_dir+'/'+'fold_'+str(fold_number)+'_'+'testing_data.csv',index=False)

def min_max_mean_auc_for_labels(results_address, labels):
    file_name=results_address+'/'+'AUC_STATS.txt'
    try:
        os.remove(file_name)
    except OSError:
        pass
    
    with open(file_name, 'a') as file:
        file.write('{:<6s}{:<13s} {:<1s} {:<1s}({:s}, {:s})\n'.format('','OUTCOMES', '|', 'AUC: avg', 'min', 'max'))
        file.write('{:<19s} {:<1s} {:s}\n'.format('-'*19, '|','-'*18))
        for label in labels:
            current_address=results_address+'/'+label
            AUROCs=pd.read_csv(current_address+'/'+'folds_AUC.csv')['AUC'].values
            minimum, maximum, mean=np.min(AUROCs), np.max(AUROCs), np.mean(AUROCs)
            if label is 'Bounceback':
                file.write('{:<5s}{:<14s} {:<1s} {:<1.2f}({:.2f}, {:.2f})\n'.format('',label, '|', mean, minimum, maximum))
            else:
                file.write('{:<19s} {:<1s} {:<1.2f}({:.2f}, {:.2f})\n'.format('readm. within '+label, '|', mean, minimum, maximum))
            file.write('{:<19s} {:<1s} {:s}\n'.format('-'*19, '|','-'*16))
        
def feature_rankings_among_all_labels_saver(current_folder,outcome_labels, conversion_tables_address):
    different_rankings=list()
    for outcome in outcome_labels:
        current_subfolder=current_folder+'/'+outcome
        ranked_feature_importances=pd.read_csv(current_subfolder+'/'+'feature_importances_sorted.csv')
        
        ranked_feature_importances.columns=['RANK','FEATURE','IMPORTANCE']
        ranked_feature_importances['RANK']=ranked_feature_importances['RANK']+1
        
        ranked_feature_importances=ranked_feature_importances[['FEATURE','RANK']]
        ranked_feature_importances.columns=['FEATURE','RANK_'+outcome]

        ranked_feature_importances=ranked_feature_importances.set_index('FEATURE')
        
        different_rankings.append(ranked_feature_importances)
        
    all_rankings=pd.concat(different_rankings, axis=1, join='inner').reset_index()
    
    all_rankings=all_rankings.set_index('FEATURE')
    all_rankings['RANKING_avg']=all_rankings.mean(axis=1)
    all_rankings['RANKING_std']=all_rankings.std(axis=1)
    
    all_rankings.to_csv(current_folder+'/'+'FEATURES_RANKING.csv')

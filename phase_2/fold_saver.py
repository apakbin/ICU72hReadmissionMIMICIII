# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:53:59 2018

@author: a.pakbin
"""
from sklearn.model_selection import StratifiedKFold
from auxiliary import grid_search,ICD9_categorizer, save_fold_data, convert_numbers_to_names, min_max_mean_auc_for_labels, train_test_one_hot_encoder, possible_values_finder,train_test_normalizer, train_test_imputer, feature_importance_saver, feature_importance_updator, save_roc_curve, data_reader, vectors_to_csv, create_subfolder_if_not_existing, feature_rankings_among_all_labels_saver
import numpy as np
import pandas as pd
from fmeasure import roc, maximize_roc
from xgboost.sklearn import XGBClassifier  
import random as rnd
from sklearn.metrics import roc_auc_score
import pickle
import gc

#TODO: __ADJUST_THESE_VALUES__
data_address="data"
file_name='df_MASTER_DATA_cleaned.csv'
writing_address='results'

#the address where MIMIC III tables are in .csv.gz format. The tables are: D_ICD_PROCEDURES.csv.gz, D_ITEMS.csv.gz and D_LABITEMS.csv.gz
conversion_tables_address='D:/Arash/MIMIC/MIMIC_DATA'
#outcome labels can contain: '24hrs' ,'48hrs','72hrs', '24hrs~72hrs','7days','30days', 'Bounceback'
outcome_labels=['24hrs']# ,'48hrs','72hrs', '24hrs~72hrs','7days','30days', 'Bounceback']
normalize_data=False
save_folds_data=False
values_for_grid_search=[np.linspace(start=1, stop=6, num=6),[50,100,200,1000,1500],[0.1]]
num_of_folds=5
#################################

categorical_column_names=['ADMISSION_TYPE', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','FIRST_CAREUNIT','GENDER']

data=data_reader(data_address, file_name)
possible_values=possible_values_finder(data, categorical_column_names)

data['IsReadmitted_24hrs~72hrs']=[1 if x>0 else 0 for x in (data['IsReadmitted_72hrs']-data['IsReadmitted_24hrs'])]

non_attribute_column_names=['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'SUBJECT_ID', 'IsReadmitted_24hrs','IsReadmitted_Bounceback','IsReadmitted_24hrs~72hrs' ,'IsReadmitted_48hrs','IsReadmitted_72hrs','IsReadmitted_7days','IsReadmitted_30days']

#TODO: for excludig insurance, language, religion, marital status and ethnicity from the data, uncomment the following line
#non_attribute_column_names=['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'SUBJECT_ID', 'IsReadmitted_24hrs','IsReadmitted_Bounceback','IsReadmitted_24hrs~72hrs' ,'IsReadmitted_48hrs','IsReadmitted_72hrs','IsReadmitted_7days','IsReadmitted_30days','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']

data=ICD9_categorizer(data)

model_type='XGB'

PREDICTIONS=list()
current_folder=writing_address

for idx, label_column_name in enumerate(['IsReadmitted_'+outcome_label for outcome_label in outcome_labels]):
    
    icu_stays=data['ICUSTAY_ID'].values
    y=data[label_column_name].values
    X=data.drop(non_attribute_column_names, axis=1)
    
    current_subfolder=current_folder+'/'+outcome_labels[idx]
    create_subfolder_if_not_existing(current_subfolder)
    
    auc_list=list()
    
    ICUstayID=list()
    Prediction=list()
    
    accumulative_feature_importance=None
    
    print ('\n',model_type, ' '*5,'LABEL: ', outcome_labels[idx])
    skf=StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=rnd.randint(1,1e6))
    fold_number=0
    for train_index, test_index in skf.split(X,y):
        fold_number+=1
        print ('\n  fold',fold_number)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        icustay_id_train, icustay_id_test=icu_stays[train_index],icu_stays[test_index]
        
        [X_TRAIN_IMPUTED, X_TEST_IMPUTED]=train_test_imputer(X_train, X_test, categorical_column_names)
        
        if normalize_data:
            [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_normalizer(X_TRAIN_IMPUTED, X_TEST_IMPUTED, categorical_column_names)  
        else:
            [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=[X_TRAIN_IMPUTED, X_TEST_IMPUTED]
            
        [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_one_hot_encoder(X_TRAIN_NORMALIZED, X_TEST_NORMALIZED, categorical_column_names, possible_values)
 
        if save_folds_data:
            save_fold_data(current_subfolder, fold_number, icustay_id_train, X_TRAIN_NORMALIZED, y_train, icustay_id_test, X_TEST_NORMALIZED, y_test, convert_names=True, conversion_tables_address=conversion_tables_address)

        [max_depths, n_estimators, learning_rates]=values_for_grid_search
        best_settings=grid_search(X=X_TRAIN_NORMALIZED, y=y_train, num_of_folds=2, verbose=True, return_auc_values=False, first_dim=max_depths, second_dim=n_estimators, third_dim=learning_rates)
        print ('{:<4s}{:<16s}: max_depth: {:<1s}, n_estimators: {:<2s}, learning_rate: {:<2s}'.format('','best hyperparameters', str(best_settings[0]), str(best_settings[1]), str(best_settings[2])))
        model=XGBClassifier(max_depth=int(best_settings[0]), n_estimators=int(best_settings[1]), learning_rate=best_settings[2])
        model.fit(X_TRAIN_NORMALIZED, y_train)
        feature_importance=model.feature_importances_
        accumulative_feature_importance=feature_importance_updator(accumulative_feature_importance, feature_importance)
    
        pd.DataFrame(data={'FEATURE_NAME': convert_numbers_to_names(X_TRAIN_NORMALIZED.columns, conversion_tables_address), 'IMPORTANCE': feature_importance}).sort_values(by='IMPORTANCE', ascending=False).reset_index(drop=True).to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_ranked_feature_importances.csv')
                     
        predictions=model.predict_proba(X_TEST_NORMALIZED)[:,1]

        ICUstayID=np.append(ICUstayID,icustay_id_test)
        Prediction=np.append(Prediction,predictions)
        
        vectors_to_csv(current_subfolder, file_name='fold_'+str(fold_number), vector_one=icustay_id_test, label_one='ICUSTAY_ID', vector_two=predictions, label_two='PREDICTION', vector_three=y_test, label_three='LABEL')

        auc=roc_auc_score(y_true=y_test, y_score=predictions)
        auc_list.append(auc)
        ROC=roc(predicted=predictions, labels=y_test)
        ROC.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc.csv')
        
        maximum=maximize_roc(ROC, maximization_criteria='fscore')
        maximum.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_optimum_point.csv')

        TPR, FPR = ROC['recall'].values, 1-ROC['specificity'] 
        
        save_roc_curve(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc_curve.jpg', TPR, FPR, auc)        
        pickle.dump(model, open(current_subfolder+'/'+'fold_'+str(fold_number)+'.model','wb'))
        print (' '+'-'*30)
    feature_importance_saver(address=current_subfolder, col_names=convert_numbers_to_names(X_TRAIN_NORMALIZED.columns, conversion_tables_address), accumulative_feature_importance=accumulative_feature_importance, num_of_folds=num_of_folds)
    vectors_to_csv(current_subfolder, file_name='folds_AUC', vector_one=auc_list, label_one='AUC', vector_two=range(1,num_of_folds+1), label_two='FOLD_NUMBER')
    gc.collect()

current_folder=writing_address
min_max_mean_auc_for_labels(current_folder, outcome_labels)
feature_rankings_among_all_labels_saver(current_folder,outcome_labels, conversion_tables_address)

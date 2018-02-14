# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:53:59 2018

@author: a.pakbin
"""
from sklearn.model_selection import StratifiedKFold
from auxiliary import grid_search,ICD9_categorizer, convert_numbers_to_names, train_test_one_hot_encoder, possible_values_finder,train_test_normalizer, train_test_imputer, feature_importance_saver, feature_importance_updator, combine_PREDICTIONs_and_save, save_roc_curve, data_reader, vectors_to_csv, create_subfolder_if_not_existing
import numpy as np
import pandas as pd
from fmeasure import roc, maximize_roc
from xgboost.sklearn import XGBClassifier  
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
import random as rnd
from sklearn.metrics import roc_auc_score
import pickle
import gc

#best_settings=[[4,100,0.1],[2,200,0.1],[3,100,0.1],[3,100,0.1]]

data_address="data"
file_name='df_MASTER_DATA_cleaned.csv'
categorical_column_names=['ADMISSION_TYPE', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','FIRST_CAREUNIT','GENDER']
categorical_column_names=['ADMISSION_TYPE','FIRST_CAREUNIT','GENDER']
#categorical_column_names=[]

data=data_reader(data_address, file_name)
possible_values=possible_values_finder(data, categorical_column_names)

data['IsReadmitted_24hrs~72hrs']=[1 if x>0 else 0 for x in (data['IsReadmitted_72hrs']-data['IsReadmitted_24hrs'])]

data['IsReadmitted_24hrs~48hrs'] =[1 if x>0 else 0 for x in (data['IsReadmitted_48hrs']-data['IsReadmitted_24hrs'])]
data['IsReadmitted_48hrs~72hrs'] =[1 if x>0 else 0 for x in (data['IsReadmitted_72hrs']-data['IsReadmitted_48hrs'])]
data['IsReadmitted_72hrs~7days'] =[1 if x>0 else 0 for x in (data['IsReadmitted_7days']-data['IsReadmitted_72hrs'])]
data['IsReadmitted_7days~30days']=[1 if x>0 else 0 for x in (data['IsReadmitted_30days']-data['IsReadmitted_7days'])]

#TODO: FOR DROPPING ADMISSIONS, PATIENTS
#TODO: ALSO UNCOMMENT LINE 23
#data=data.drop(['FIRST_CAREUNIT','LOS','ADMISSION_TYPE','INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY','GENDER','AGE'], axis=1)

#TODO: FOR DROPPING PROCEDURES
#data=data.drop([col for col in data.columns if str(col).startswith('PROCEDURE_')],axis=1)

#TODO: FOR DROPPING ICD's
#data=data.drop([col for col in data.columns if str(col).startswith('ICD9_')],axis=1)

#outcome_labels=['72hrs',
outcome_labels=['24hrs~48hrs','48hrs~72hrs','72hrs~7days','7days~30days']
#outcome_labels=['24hrs']
#outcome_labels=['7days','30days','Bounceback']
non_attribute_column_names=['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'SUBJECT_ID', 'IsReadmitted_24hrs','IsReadmitted_Bounceback','IsReadmitted_24hrs~72hrs' ,'IsReadmitted_48hrs','IsReadmitted_72hrs','IsReadmitted_7days','IsReadmitted_30days']
non_attribute_column_names=['IsReadmitted_7days~30days','IsReadmitted_72hrs~7days','IsReadmitted_48hrs~72hrs','IsReadmitted_24hrs~48hrs','HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'SUBJECT_ID', 'IsReadmitted_24hrs','IsReadmitted_Bounceback','IsReadmitted_24hrs~72hrs' ,'IsReadmitted_48hrs','IsReadmitted_72hrs','IsReadmitted_7days','IsReadmitted_30days','INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']

data=ICD9_categorizer(data)

model_types=['XGB']

for model_type in model_types:
    PREDICTIONS=list()
    current_folder=data_address+'/'+'NOT_BCKWD_INCLUSIVE'+'/'+model_type
    
    for idx, label_column_name in enumerate(['IsReadmitted_'+outcome_label for outcome_label in outcome_labels]):
        
        icu_stays=data['ICUSTAY_ID'].values
        y=data[label_column_name].values
        X=data.drop(non_attribute_column_names, axis=1)
        
        current_subfolder=current_folder+'/'+outcome_labels[idx]
        create_subfolder_if_not_existing(current_subfolder)
        
        auc_list=list()
        num_of_folds=5
        
        ICUstayID=list()
        Prediction=list()
        
        accumulative_feature_importance=None
        
        skf=StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=rnd.randint(1,1e6))
        fold_number=0
        for train_index, test_index in skf.split(X,y):
            fold_number+=1
            print ('\n',model_type,': ',outcome_labels[idx],fold_number)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            icustay_id_train, icustay_id_test=icu_stays[train_index],icu_stays[test_index]
            
            [X_TRAIN_IMPUTED, X_TEST_IMPUTED]=train_test_imputer(X_train, X_test, categorical_column_names)
            if model_type!='XGB':
                [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_normalizer(X_TRAIN_IMPUTED, X_TEST_IMPUTED, categorical_column_names)  
            else:
                [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=[X_TRAIN_IMPUTED, X_TEST_IMPUTED]
                
            [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_one_hot_encoder(X_TRAIN_NORMALIZED, X_TEST_NORMALIZED, categorical_column_names, possible_values)
            
            if model_type=='XGB':
                max_depths=np.linspace(start=1, stop=6, num=6)
                n_estimators=[50,100,200,1000,1500]
                learning_rates=[0.1]
                best_settings=grid_search(model_type='XGB', X=X_TRAIN_NORMALIZED, y=y_train, num_of_folds=2, verbose=True, return_auc_values=False, first_dim=max_depths, second_dim=n_estimators, third_dim=learning_rates)
                #best_settings=[6, 2000, 0.1]
                print ('best setting: ', best_settings)
                model=XGBClassifier(max_depth=int(best_settings[0]), n_estimators=int(best_settings[1]), learning_rate=best_settings[2])
                model.fit(X_TRAIN_NORMALIZED, y_train)
                feature_importance=model.feature_importances_
                accumulative_feature_importance=feature_importance_updator(accumulative_feature_importance, feature_importance)
            
                pd.DataFrame(data={'col_name': X_TRAIN_NORMALIZED.columns, 'importance': feature_importance}).sort_values(by='importance', ascending=False).reset_index(drop=True).to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_ranked_feature_importances.csv')
            
            if model_type=='LR':
                model=LogisticRegressionCV(penalty='l1', solver='saga', n_jobs=-1, cv=5, verbose=False)
                model.fit(X_TRAIN_NORMALIZED, y_train)
    
            if model_type=='SGD':
                alphas=np.logspace(start=-10, stop=3, num=14)
                best_setting=grid_search(model_type='SGD', X=X_TRAIN_NORMALIZED, y=y_train, num_of_folds=5, verbose=True, return_auc_values=False, first_dim=alphas)
                model=SGDClassifier(penalty='l1', loss='log', alpha=best_setting, shuffle=True, class_weight='balanced', n_jobs=-1)
                model.fit(X_TRAIN_NORMALIZED, y_train)             
    
            predictions=model.predict_proba(X_TEST_NORMALIZED)[:,1]
    
            ICUstayID=np.append(ICUstayID,icustay_id_test)
            Prediction=np.append(Prediction,predictions)
            
            vectors_to_csv(current_subfolder, file_name='fold_'+str(fold_number), vector_one=icustay_id_test, label_one='icustay_id', vector_two=predictions, label_two='prediction', vector_three=y_test, label_three='label')
    
            auc=roc_auc_score(y_true=y_test, y_score=predictions)
            auc_list.append(auc)
            ROC=roc(predicted=predictions, labels=y_test)
            ROC.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc.csv')
            
            maximum=maximize_roc(ROC, maximization_criteria='fscore')
            maximum.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_optimum_point.csv')
    
            TPR, FPR = ROC['recall'].values, 1-ROC['specificity'] 
            
            save_roc_curve(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc_curve.jpg', TPR, FPR, auc)        
            pickle.dump(model, open(current_subfolder+'/'+'fold_'+str(fold_number)+'_model.model','wb'))
        
        PREDICTION=pd.DataFrame(data={'icustay_id':ICUstayID, outcome_labels[idx]:Prediction}).set_index('icustay_id')
        PREDICTIONS.append(PREDICTION)
        
        if model_type=='XGB':
            feature_importance_saver(address=current_subfolder, col_names=convert_numbers_to_names(X_TRAIN_NORMALIZED.columns), accumulative_feature_importance=accumulative_feature_importance, num_of_folds=num_of_folds)
        vectors_to_csv(current_subfolder, file_name='folds_AUC', vector_one=auc_list, label_one='AUC', vector_two=range(1,num_of_folds+1), label_two='fold_number')
        #print ('DONE: label:'+outcome_labels[idx])
        gc.collect()
    
    combine_PREDICTIONs_and_save(current_folder, PREDICTIONS)
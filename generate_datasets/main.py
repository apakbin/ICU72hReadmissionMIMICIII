# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 14:23:18 2018

@author: rafip
"""

import yaml
import psycopg2
from os.path import join
import pandas as pd
import numpy as np
from utils import SQLConnection, clean_nan_columns

pd.set_option('mode.chained_assignment', None)

def getfeaturesFromStaticTables(config):
    print('generating features from non-events tables...')

    data_dir = config['data_dir']

    #get relevant data from ADMISSIONS table
    print('\nImporting data from ADMISSIONS...')
    path_admissions = join(data_dir, 'ADMISSIONS.csv.gz')
    df_admissions = pd.read_csv(path_admissions)
    trimmed_admissions = df_admissions[
        ['HADM_ID', 'SUBJECT_ID', 'ADMISSION_TYPE', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']]

    # get relevant data from ICUSTAYS table
    print('\nImporting data from ICUSTAYS...')
    path_icustays = join(data_dir, 'ICUSTAYS.csv.gz')
    df_icustays= pd.read_csv(path_icustays)
    trimmed_icustays = df_icustays[['HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT', 'LOS', 'INTIME', 'OUTTIME']]

    # get relevant data from PATIENTS table
    print('\nImporting data from PATIENTS...')
    path_patients = join(data_dir, 'PATIENTS.csv.gz')
    df_patients = pd.read_csv(path_patients)
    trimmed_patients = df_patients[['SUBJECT_ID', 'GENDER', 'DOB']]

    # get relevant data from PROCEDUREEVENTS_MV table
    print('\nImporting data from PROCEDUREEVENTS_MV...')
    path_procedures = join(data_dir, 'PROCEDUREEVENTS_MV.csv.gz')
    df_procedures = pd.read_csv(path_procedures)
    trimmed_procedures = df_procedures[['ICUSTAY_ID', 'ORDERCATEGORYNAME']]

    trimmed_procedures['DUMMY'] = pd.Series([x for x in range(0, len(df_procedures))])
    pivoted_procedures = trimmed_procedures.pivot_table(index='ICUSTAY_ID', columns='ORDERCATEGORYNAME', values='DUMMY',
                                                        fill_value=0).astype('bool').astype('int')
    pivoted_procedures.columns = ['PROCEDURE_' + str(col_name) for col_name in pivoted_procedures.columns]
    pivoted_procedures = pivoted_procedures.reset_index()

    # get relevant data from PROCEDURES_ICD table
    print('\nImporting data from PROCEDURES_ICD...')
    path_procedures = join(data_dir, 'PROCEDURES_ICD.csv.gz')
    df_ICD9 = pd.read_csv(path_procedures)
    trimmed_ICD9 = df_ICD9[['HADM_ID','SEQ_NUM','ICD9_CODE']]
    pivoted_ICD9 = trimmed_ICD9.pivot_table(index='HADM_ID', columns='ICD9_CODE', values='SEQ_NUM', fill_value=0).astype(
        'bool').astype('int')
    pivoted_ICD9.columns = ['ICD9_' + str(col_name) for col_name in pivoted_ICD9.columns]
    pivoted_ICD9 = pivoted_ICD9.reset_index()

    # merging dataframes
    print('\nMerging data from ADMISSIONS, ICUSTAYS, PATIENTS, PROCEDUREEVENTS_MV, PROCEDURES_ICD...')
    df_merged = trimmed_icustays.merge(trimmed_admissions, on='HADM_ID', how='left')
    df_merged = df_merged.merge(trimmed_patients, on='SUBJECT_ID', how='left')
    df_merged = df_merged.merge(pivoted_procedures, on='ICUSTAY_ID', how='left')
    df_merged = df_merged.merge(pivoted_procedures, on='ICUSTAY_ID',  how='left')
    df_merged = df_merged.merge(pivoted_ICD9, on='HADM_ID',how='left')

    # Calculating age and median correcting deidentified ages of ovelrly aged people
    ages = (df_merged['INTIME'].astype('datetime64[ns]') - df_merged['DOB'].astype('datetime64[ns]')).dt.days / 365
    df_merged['AGE'] = [age if age >= 0 else 91.4 for age in ages]
    df_merged.drop(['DOB'], axis=1, inplace=True)

    #Fixing missing values in PROCEDURE and ICD9 columns
    NAN_MEANS_NO_COLUMNS_PREFIXES = list()
    NAN_MEANS_NO_COLUMNS_PREFIXES.append('PROCEDURE')
    NAN_MEANS_NO_COLUMNS_PREFIXES.append('ICD9')

    for col_name in df_merged.columns:
        for nan_means_no_column_prefix in NAN_MEANS_NO_COLUMNS_PREFIXES:
            if col_name.startswith(nan_means_no_column_prefix):
                df_merged[col_name].fillna(0, inplace=True)

    return df_merged


def getfeaturesFromEventsTables(config):

    print('generating features from labevents and chartevents...')

    conn = SQLConnection(config)

    #features from labevents
    query_labslastmsmts = '''
        select * from labslastmsmts;
        '''
    df_labslastmsmts = conn.executeQuery(query_labslastmsmts)
    df_labslastmsmts.drop('max_charttime', axis=1, inplace=True)

    df_labslastmsmts = df_labslastmsmts.pivot_table(index='icustay_id', columns='itemid', values='valuenum', aggfunc='mean')
    new_cols = [str(col) + '_lastmsrmt' for col in df_labslastmsmts.columns]
    df_labslastmsmts.columns = new_cols
    df_labslastmsmts = clean_nan_columns(df_labslastmsmts, thres = 60)

    #features from labevents
    query_chartslastmsmts = '''
        select * from chartslastmsmts;
        '''
    df_chartslastmsmts = conn.executeQuery(query_chartslastmsmts)
    df_chartslastmsmts = df_chartslastmsmts[df_chartslastmsmts.itemid.isnull() == False]
    df_chartslastmsmts['itemid'] = df_chartslastmsmts['itemid'].astype('int')
    df_chartslastmsmts = df_chartslastmsmts.pivot_table(index='icustay_id', columns='itemid', values='lastmsmt',aggfunc='mean')
    df_chartslastmsmts.reset_index(inplace=True)

    return df_labslastmsmts


if __name__ == "__main__":
    config = yaml.safe_load(open("../resources/config.yml"))
    #df_static_tables = getfeaturesFromStaticTables(config=config)
    df_events_tables = getfeaturesFromEventsTables(config=config)
    print(df_events_tables.head())
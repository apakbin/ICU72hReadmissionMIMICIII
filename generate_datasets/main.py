# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 14:23:18 2018

@author: rafip
"""

import yaml
from os.path import join
import pandas as pd
import numpy as np
from utils import SQLConnection, clean_nan_columns, get_features_from_labevents, get_features_from_chartevents

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
    # removing minors from the data
    df_merged = df_merged[(df_merged['AGE']>= 18)]

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
    conn = SQLConnection(config)
    
    # getting features from labevents
    query_labmsmts = '''
            select * from LABMEASURMENTS;
            '''
    df_labmsmts = conn.executeQuery(query_labmsmts)

    #observations from labevents at discharge time
    query_labslastmsmts = '''
        select * from labslastmsmts;
        '''
    df_labslastmsmts = conn.executeQuery(query_labslastmsmts)

    #all extacted features from labevents
    df_labs_features = get_features_from_labevents(df_labmsmts, df_labslastmsmts)

    # getting features from chartevents
    query_chartsmsmts = '''
            select * from CHARTSMEASURMENTS;
            '''
    df_chartsmsmts = conn.executeQuery(query_chartsmsmts)

    # getting features from chartevents
    query_chartslastmsmts = '''
            select * from CHARTSLASTMSMTS;
            '''
    df_chartslastmsmts = conn.executeQuery(query_chartslastmsmts)
    df_chart_features = get_features_from_chartevents(df_chartsmsmts, df_chartslastmsmts)
    df_events_tables = pd.merge(df_labs_features, df_chart_features, how='left', on='icustay_id')
    return df_events_tables

def getfeaturesFromSeverityScoreConcepts(config):
    """
    The scripts generate features from MIMIC-III concepts tables for severity scores
    :param config:
    :return:
    """
    print("\n Extracting features from MIMIC-III concepts tables for severity scores")
    conn = SQLConnection(config)
    sapsii = conn.executeQuery('''select * from sapsii;''')
    sapsii = sapsii[sapsii.icustay_id.isnull() == False]
    sapsii['icustay_id'] = sapsii['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    sapsii.drop(cols_to_remove, axis=1, inplace=True)

    sofa = conn.executeQuery('''select * from sofa;''')
    sofa = sofa[sofa.icustay_id.isnull() == False]
    sofa['icustay_id'] = sofa['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    sofa.drop(cols_to_remove, axis=1, inplace=True)

    sirs = conn.executeQuery('''select * from sirs;''')
    sirs = sirs[sirs.icustay_id.isnull() == False]
    sirs['icustay_id'] = sirs['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    sirs.drop(cols_to_remove, axis=1, inplace=True)

    lods = conn.executeQuery('''select * from lods;''')
    lods = lods[lods.icustay_id.isnull() == False]
    lods['icustay_id'] = lods['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    lods.drop(cols_to_remove, axis=1, inplace=True)

    apsiii = conn.executeQuery('''select * from apsiii;''')
    apsiii = apsiii[apsiii.icustay_id.isnull() == False]
    apsiii['icustay_id'] = apsiii['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    apsiii.drop(cols_to_remove, axis=1, inplace=True)

    oasis = conn.executeQuery('''select * from oasis;''')
    oasis = oasis[oasis.icustay_id.isnull() == False]
    oasis['icustay_id'] = oasis['icustay_id'].astype('int')
    cols_to_remove = ['subject_id', 'hadm_id']
    oasis.drop(cols_to_remove, axis=1, inplace=True)

    sapsii_sofa = pd.merge(sapsii, sofa, on='icustay_id')
    # removing repeated column
    sapsii_sofa.drop(['temp_score'], axis=1, inplace=True)
    sapsii_sofa_sirs = pd.merge(sapsii_sofa, sirs, on='icustay_id')
    # removing repeated column
    sapsii_sofa_sirs.drop(['cardiovascular'], axis=1, inplace=True)
    sapsii_sofa_sirs_lods = pd.merge(sapsii_sofa_sirs, lods, on='icustay_id')
    sapsii_sofa_sirs_lods_apsiii = pd.merge(sapsii_sofa_sirs_lods, apsiii, on='icustay_id')
    df_severity_scores = pd.merge(sapsii_sofa_sirs_lods_apsiii, oasis, on='icustay_id')

    # filling score columns with 0 since in mimic db mean scores is taken as 0
    score_columns = df_severity_scores.filter(regex='score').columns
    df_severity_scores[score_columns] = df_severity_scores[score_columns].fillna(0)

    # removing columns with > 50% NA value
    df_severity_scores = clean_nan_columns(df_severity_scores, thres=60)
    # cleaning up non-numeric columns from severity scores
    df_severity_scores.drop(['icustay_age_group', 'preiculos'], axis=1, inplace=True)

    return df_severity_scores

def addLOSFeature(df_MASTER_DATA):
    # remove already existing LOS features
    df_MASTER_DATA.drop(['LOS'], axis=1, inplace=True)
    
    df_MASTER_DATA = df_MASTER_DATA[df_MASTER_DATA.OUTTIME.isnull() == False]
    df_MASTER_DATA = df_MASTER_DATA[df_MASTER_DATA.INTIME.isnull() == False]

    df_MASTER_DATA['INTIME'] = df_MASTER_DATA['INTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['OUTTIME'] = df_MASTER_DATA['OUTTIME'].astype('datetime64[ns]')
    
    df_MASTER_DATA.reset_index(drop=True, inplace=True)
    
    # Add new los feature
    los = []
    short_los = []
    num_adms = df_MASTER_DATA.shape[0]
    for idx in range(num_adms):
        los_hours = (df_MASTER_DATA.OUTTIME[idx] - df_MASTER_DATA.INTIME[idx]).seconds / 3600 + \
                    (df_MASTER_DATA.OUTTIME[idx] - df_MASTER_DATA.INTIME[idx]).days * 24
        los_days = los_hours / 24
        los.append(los_days)

        # if LOS less than median LOS
        if los_days < 2.144537:
            short_los.append(1)
        else:
            short_los.append(0)

    df_MASTER_DATA['LOS'] = los
    df_MASTER_DATA['Is_Short_LOS'] = short_los
    df_MASTER_DATA['LOS'] = df_MASTER_DATA['LOS'].apply(lambda x: np.log(x))
    return df_MASTER_DATA


def addTargetFeatures(df_MASTER_DATA):
    """
    the function adds target labels to the dataset
    :param df_MASTER_DATA:
    :return:
    """
    print('\n Adding target variables...')

    df_MASTER_DATA = df_MASTER_DATA[df_MASTER_DATA.OUTTIME.isnull() == False]
    df_MASTER_DATA = df_MASTER_DATA[df_MASTER_DATA.INTIME.isnull() == False]

    df_MASTER_DATA['INTIME'] = df_MASTER_DATA['INTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['OUTTIME'] = df_MASTER_DATA['OUTTIME'].astype('datetime64[ns]')
    df_MASTER_DATA = df_MASTER_DATA.sort_values(['SUBJECT_ID', 'INTIME', 'OUTTIME'], ascending=[True, True, True])
    df_MASTER_DATA.reset_index(inplace=True, drop=True)

    # Add targetr column to show if readmitted within different timeframes
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_24hrs=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_48hrs=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_72hrs=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_7days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_30days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_Bounceback=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(Time_To_readmission=np.nan)

    # total number of admissions
    num_adms = df_MASTER_DATA.shape[0]

    for idx in range(1, num_adms):
        if df_MASTER_DATA.SUBJECT_ID[idx] == df_MASTER_DATA.SUBJECT_ID[idx - 1]:
            # previous icu discharge time
            prev_outtime = df_MASTER_DATA.OUTTIME[idx - 1]
            # current icu admit time
            curr_intime = df_MASTER_DATA.INTIME[idx]

            readmit_time = curr_intime - prev_outtime
            df_MASTER_DATA.loc[idx - 1, 'Time_To_readmission'] = readmit_time.seconds / (
                    3600 * 24) + readmit_time.days

            if readmit_time.days <= 1:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_24hrs'] = 1
            if readmit_time.days <= 2:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_48hrs'] = 1
            if readmit_time.days <= 3:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_72hrs'] = 1
            if readmit_time.days <= 7:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_7days'] = 1
            if readmit_time.days <= 30:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_30days'] = 1
                # Check bouncebacks within 30 days
                if df_MASTER_DATA.HADM_ID[idx] == df_MASTER_DATA.HADM_ID[idx - 1]:
                    df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_Bounceback'] = 1



if __name__ == "__main__":
    config = yaml.safe_load(open("../resources/config.yml"))
    df_static_tables = getfeaturesFromStaticTables(config=config)
    df_events_tables = getfeaturesFromEventsTables(config=config)
    df_severity_scores = getfeaturesFromSeverityScoreConcepts(config=config)

    print('\n Merging static features, events features and severity scores')
    df_MASTER_DATA = pd.merge(df_static_tables,df_events_tables, how='left', left_on='ICUSTAY_ID', right_on ='icustay_id', left_index=True, right_index=True)
    df_MASTER_DATA.drop(['icustay_id'], axis=1, inplace=True)
    df_MASTER_DATA = pd.merge(df_MASTER_DATA, df_severity_scores, how='left', left_on='ICUSTAY_ID', right_on='icustay_id')
    df_MASTER_DATA.drop(['icustay_id'], axis=1, inplace=True)
    df_MASTER_DATA = addLOSFeature(df_MASTER_DATA)

    #Cleaning up outliers
    # Fixing Heart rates
    HR_columns_to_fix = df_MASTER_DATA.filter(regex='^Heart_Rate((?!count).)*$').columns
    df_MASTER_DATA[HR_columns_to_fix] = df_MASTER_DATA[HR_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing Arterial_BP_Diastolic
    ABPD_columns_to_fix = df_MASTER_DATA.filter(regex='^Arterial_BP_Diastolic((?!count).)*$').columns
    df_MASTER_DATA[ABPD_columns_to_fix] = df_MASTER_DATA[ABPD_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing Arterial_BP_Systolic
    ABPS_columns_to_fix = df_MASTER_DATA.filter(regex='^Arterial_BP_Systolic((?!count).)*$').columns
    df_MASTER_DATA[ABPS_columns_to_fix] = df_MASTER_DATA[ABPS_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing Mean_Arterial_BP
    MABP_columns_to_fix = df_MASTER_DATA.filter(regex='^Mean_Arterial_BP((?!count).)*$').columns
    df_MASTER_DATA[MABP_columns_to_fix] = df_MASTER_DATA[MABP_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing NBP_Systolic
    NBPS_columns_to_fix = df_MASTER_DATA.filter(regex='^NBP_Systolic((?!count).)*$').columns
    df_MASTER_DATA[NBPS_columns_to_fix] = df_MASTER_DATA[NBPS_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing NBP_Diastolic
    NBPD_columns_to_fix = df_MASTER_DATA.filter(regex='^NBP_Diastolic((?!count).)*$').columns
    df_MASTER_DATA[NBPD_columns_to_fix] = df_MASTER_DATA[NBPD_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing NBP_Mean
    MNBP_columns_to_fix = df_MASTER_DATA.filter(regex='^Mean_NBP((?!count).)*$').columns
    df_MASTER_DATA[MNBP_columns_to_fix] = df_MASTER_DATA[MNBP_columns_to_fix].apply(
        lambda x: [None if y > 300 or y < 0 else y for y in x])

    # Fixing Temperature
    Temp_columns_to_fix = df_MASTER_DATA.filter(regex='^Temperature((?!count).)*$').columns
    df_MASTER_DATA[Temp_columns_to_fix] = df_MASTER_DATA[Temp_columns_to_fix].apply(
        lambda x: [None if y > 114 or y < 77 else y for y in x])

    # Fixing spO2
    spo2_columns_to_fix = df_MASTER_DATA.filter(regex='^SpO2((?!count).)*$').columns
    df_MASTER_DATA[spo2_columns_to_fix] = df_MASTER_DATA[spo2_columns_to_fix].apply(
        lambda x: [None if y > 100 or y < 0 else y for y in x])

    # Fixing Weight
    WT_columns_to_fix = df_MASTER_DATA.filter(regex='^Weight((?!count).)*$').columns
    df_MASTER_DATA[WT_columns_to_fix] = df_MASTER_DATA[WT_columns_to_fix].apply(
        lambda x: [None if y > 500 or y < 30 else y for y in x])

    df_MASTER_DATA = addTargetFeatures(df_MASTER_DATA)
    data_dir = config['data_dir']
    datasetPath = join(data_dir, 'df_MASTER_DATA.csv')
    df_MASTER_DATA.to_csv(datasetPath, index=False)

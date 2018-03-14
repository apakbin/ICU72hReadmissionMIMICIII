import psycopg2
import pandas as pd

class SQLConnection(object):
    def __init__(self, config):
        self.dbname = config['dbname']
        self.user = config['user']
        self.host = config['host']
        self.port = config['port']
        self.password = config['password']
        self.schema = config['schema']

    def executeQuery(self, query):
        con = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, port=self.port,password=self.password)
        cur = con.cursor()
        cur.execute('SET search_path to {}'.format(self.schema))
        results = pd.read_sql_query(query, con)
        cur.close()
        con.close()
        return results


def clean_nan_columns(df, thres):
    """
    The function removes columns containing missing values more than threshold
    :param df:
    :param thres:
    :return:
    """
    nan_cols = df.isnull().sum() * 100 / df.shape[0]
    nan_cols_sorted = nan_cols.sort_values(ascending=False)
    df_cleaned = df.drop(list(nan_cols_sorted[nan_cols_sorted > 60].index), axis=1)
    return df_cleaned


def get_features_from_labevents(df_labmsmts, df_labslastmsmts):
    """
    The function extracts features from labevents
    :param df_labmsmts:
    :param df_labslastmsmts:
    :return:
    """
    print('generating features from labevents...')

    df5 = df_labmsmts.pivot(index='icustay_id', columns='itemid', values='mean_val')
    df6 = df_labmsmts.pivot(index='icustay_id', columns='itemid', values='max_val')
    df7 = df_labmsmts.pivot(index='icustay_id', columns='itemid', values='min_val')
    df8 = df_labmsmts.pivot(index='icustay_id', columns='itemid', values='stddev_val')
    df9 = df_labmsmts.pivot(index='icustay_id', columns='itemid', values='count_val')
    df5.columns = [str(col) + '_mean' for col in df5.columns]
    df6.columns = [str(col) + '_max' for col in df6.columns]
    df7.columns = [str(col) + '_min' for col in df7.columns]
    df8.columns = [str(col) + '_stddev' for col in df8.columns]
    df9.columns = [str(col) + '_count' for col in df9.columns]

    df5 = df5.reset_index()
    df6 = df6.reset_index()
    df7 = df7.reset_index()
    df8 = df8.reset_index()
    df9 = df9.reset_index()

    df_labs_features = pd.merge(df5, df6, on='icustay_id')
    df_labs_features = pd.merge(df_labs_features, df7, on='icustay_id')
    df_labs_features = pd.merge(df_labs_features, df8, on='icustay_id')
    df_labs_features = pd.merge(df_labs_features, df9, on='icustay_id')

    # remove missing values
    df_labs_features = clean_nan_columns(df_labs_features, thres=50)

    df_labslastmsmts.drop('max_charttime', axis=1, inplace=True)
    df_labslastmsmts = df_labslastmsmts.pivot_table(index='icustay_id', columns='itemid', values='valuenum',
                                                    aggfunc='mean')
    new_cols = [str(col) + '_lastmsrmt' for col in df_labslastmsmts.columns]
    df_labslastmsmts.columns = new_cols
    df_labslastmsmts = clean_nan_columns(df_labslastmsmts, thres=60)
    df_labslastmsmts.reset_index(inplace=True)

    df_labs_features = pd.merge(df_labs_features, df_labslastmsmts, on='icustay_id')

    return df_labs_features

def merge_duplicate_features(df_chart_features, df_chart_master, feature_types,new_feature,feature_ids):
    """
    function to merge duplicate features in chartevents
    :param df_chart_features:
    :param df_chart_master:
    :param feature_types:
    :param new_feature:
    :param feature_ids:
    :return:
    """
    for feature_type in feature_types:
        discrete_cols_per_type = [str(feature_id) + '_' + feature_type for feature_id in feature_ids]

        new_col = new_feature + '_' + feature_type
        if feature_type == 'count':
            df_chart_features[new_col] = df_chart_master[discrete_cols_per_type].max(axis=1)
            df_chart_features[new_col].fillna(value=0, inplace=True)
        else:
            df_chart_features[new_col] = df_chart_master[discrete_cols_per_type].mean(axis=1)
        return df_chart_features


def get_features_from_chartevents(df_chartsmsmts, df_chartslastmsmts):
    print('generating features from chartevents...')
    df1 = df_chartsmsmts.pivot(index='icustay_id', columns='itemid', values='mean_val')
    df2 = df_chartsmsmts.pivot(index='icustay_id', columns='itemid', values='max_val')
    df3 = df_chartsmsmts.pivot(index='icustay_id', columns='itemid', values='min_val')
    df4 = df_chartsmsmts.pivot(index='icustay_id', columns='itemid', values='stddev_val')
    df5 = df_chartsmsmts.pivot(index='icustay_id', columns='itemid', values='count_val')

    df1.columns = [str(col) + '_mean' for col in df1.columns]
    df2.columns = [str(col) + '_max' for col in df2.columns]
    df3.columns = [str(col) + '_min' for col in df3.columns]
    df4.columns = [str(col) + '_stddev' for col in df4.columns]
    df5.columns = [str(col) + '_count' for col in df5.columns]

    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df3 = df3.reset_index()
    df4 = df4.reset_index()
    df5 = df5.reset_index()

    df_charts_master = pd.merge(df1, df2, on='icustay_id')
    df_charts_master = pd.merge(df_charts_master, df3, on='icustay_id')
    df_charts_master = pd.merge(df_charts_master, df4, on='icustay_id')
    df_charts_master = pd.merge(df_charts_master, df5, on='icustay_id')

    #Many features in chartevents were duplicate
    #Following code merges duplicate columns to create new features
    ## Create new dataframe
    df_chart_features = pd.DataFrame()
    df_chart_features['icustay_id'] = df_charts_master['icustay_id']

    #Creating Features for Heart Rate
    feature_ids = [211, 220045]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Heart_Rate'
    df_chart_features = merge_duplicate_features(df_chart_features,df_charts_master, feature_types, new_feature, feature_ids)

    #Creating Features for Weight
    # Convert weight kgs to pounds
    def convert_weight(x):
        x = x * 2.2
        return float(x)

    feature_types = ['mean', 'max', 'min', 'stddev']
    kg_cols = [3581, 226531]

    for ftr_type in feature_types:
        for kg_col in kg_cols:
            df_charts_master[str(kg_col) + '_' + ftr_type] = df_charts_master[str(kg_col) + '_' + ftr_type].apply(convert_weight)

    feature_ids = [763, 3581, 3583, 226512, 226531, 3693]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Weight'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature, feature_ids)

    #Creating Features for Respiratory Rate
    feature_ids = [618, 220210]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Respiratory_Rate'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature, feature_ids)

    #Creating Features for Blood Pressures
    feature_ids = [51, 220050, 6, 6701, 225309]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Arterial_BP_Systolic'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    feature_ids = [8364, 8368, 8555, 220051, 225310]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Arterial_BP_Diastolic'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    feature_ids = [224, 52, 6702, 6927, 220052]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Mean_Arterial_BP'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    feature_ids = [455, 220179]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'NBP_Systolic'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    feature_ids = [8441, 220180]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'NBP_Diastolic'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    feature_ids = [456, 220181]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Mean_NBP'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    #Creating Features for Temperature
    # Convert Celcius to Fahrenheit
    def convert_Temp(x):
        x = x * 1.8 + 32
        return float(x)

    feature_types = ['mean', 'max', 'min', 'stddev']
    celcius_cols = [676, 223762]
    for ftr_type in feature_types:
        for cel_cols in celcius_cols:
            df_charts_master[str(cel_cols) + '_' + ftr_type] = df_charts_master[str(cel_cols) + '_' + ftr_type].apply(convert_Temp)


    feature_ids = [676, 678, 223761, 223762]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'Temperature'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)

    #Creating features for sp02
    feature_ids = [220277, 646]
    feature_types = ['mean', 'max', 'min', 'count', 'stddev']
    new_feature = 'SpO2'
    df_chart_features = merge_duplicate_features(df_chart_features, df_charts_master, feature_types, new_feature,
                                                 feature_ids)


    #Features from chartevents a the time of discharge
    df_chartslastmsmts = df_chartslastmsmts[df_chartslastmsmts.itemid.isnull() == False]
    df_chartslastmsmts['itemid'] = df_chartslastmsmts['itemid'].astype('int')

    df_chartslastmsmts = df_chartslastmsmts.pivot_table(index='icustay_id', columns='itemid', values='lastmsmt',
                                                  aggfunc='mean')
    df_chartslastmsmts.reset_index(inplace=True)

    # create last measured feature for temp
    for cel_cols in celcius_cols:
        df_chartslastmsmts[cel_cols] = df_chartslastmsmts[cel_cols].apply(convert_Temp)

    feature_ids = [676, 678, 223761, 223762]
    new_feature = 'Temperature'
    new_col = new_feature + '_lastmsrmt'
    df_chart_features[new_col] = df_chartslastmsmts[feature_ids].mean(axis=1)

    # create last measured feature for HeartRate
    feature_ids = [211, 220045]
    new_feature = 'Heart_Rate'
    new_col = new_feature + '_lastmsrmt'
    df_chart_features[new_col] = df_chartslastmsmts[feature_ids].mean(axis=1)

    # create last measured feature for Respiratory Rate
    feature_ids = [618, 220210]
    new_feature = 'Respiratory_Rate'
    new_col = new_feature + '_lastmsrmt'
    df_chart_features[new_col] = df_chartslastmsmts[feature_ids].mean(axis=1)

    return df_chart_features

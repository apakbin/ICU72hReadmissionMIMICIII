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
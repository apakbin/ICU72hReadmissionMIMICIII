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
        return results
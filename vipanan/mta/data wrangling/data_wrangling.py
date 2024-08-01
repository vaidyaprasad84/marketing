import pandas as pd
import numpy as np
from datetime import datetime,timedelta

class random_dataframe:
    def __init__(self):
        pass
    
    def create_random_df(self,
                         seed=None,
                      nrows=10000,
                      channels = ['FB','X','Email','Direct','Google'], 
                      channel_weights = [0.2]*5,
                      conversion_rate = 0.1,
                      start_date = None,
                      delta = None):
        np.random.seed(seed)

        user_id = np.random.choice(2000,nrows)
        chan_list = np.random.choice(len(channels),nrows,p=channel_weights)
        conv = [0,1]
        conv_weights = [1-conversion_rate,conversion_rate]
        conversions = np.random.choice(2,nrows,p=conv_weights)
        today = datetime.today()
        delta_dates = list(np.random.choice(30,nrows))
        dates_list = [today + timedelta(days = -int(x)) for x in delta_dates]
        df = pd.DataFrame({'user_id':user_id,
                          'channel_id':chan_list,
                          'event_date':dates_list,
                          'conversion':conversions})
        chan_df = pd.DataFrame({'channel':channels,'channel_id':list(range(len(channels)))})
        df = pd.merge(df,chan_df,on = 'channel_id',how = 'inner')
        df = df[['user_id','channel','event_date','conversion']]
        df = df.sort_values(by = ['user_id','event_date']).reset_index().drop(columns = ['index'])
        conv_df = df[df['conversion']==1]
        conv_df = pd.DataFrame(conv_df.groupby(['user_id'])['event_date'].max()).reset_index()
        conv_df['conv'] = 1
        df = pd.merge(df,conv_df, on = ['user_id','event_date'], how = 'left')
        df = df.fillna(0)
        df = df.drop(columns = ['conversion']).rename(columns = {'conv':'conversion'})
        conv_df = df[df['conversion']==1]
        conv_df = conv_df.groupby(['user_id','event_date'])['channel'].max().reset_index()
        conv_df['conv'] = 1
        df = pd.merge(df,conv_df, on = ['user_id','event_date','channel'], how = 'left')
        df = df.fillna(0)
        df = df.drop(columns = ['conversion']).rename(columns = {'conv':'conversion'})
        df['conversion'] = df['conversion'].astype('int')
        df = df.drop_duplicates().reset_index().drop(columns = ['index'])
        return df
    
random_dataframe = random_dataframe()

class attribution_df:
    def __init__(self):
        pass
    
    def get_data_in_attr_window(self,
                                attribution_window = 30,
                                path_transforms = 'unique',
                                dataframe = pd.DataFrame({})
                               ):
        
        filter_df = dataframe[dataframe['conversion']==1][['user_id','event_date']].rename(columns = {'event_date':'conv_date'})
        dataframe = pd.merge(dataframe,filter_df,on = 'user_id',how = 'outer')
        dataframe['conv_date'] = dataframe['conv_date'].fillna(dataframe['event_date'])
        dataframe['delta'] = (dataframe['conv_date'] - dataframe['event_date']).dt.days
        dataframe = dataframe[((dataframe['delta']<=attribution_window) & 
                              (dataframe['delta'] >=0))].drop(columns =['conv_date','delta'])
        return dataframe
    
attribution_df = attribution_df()


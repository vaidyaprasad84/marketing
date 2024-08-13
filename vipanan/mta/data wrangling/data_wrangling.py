import pandas as pd
import numpy as np
from datetime import datetime,timedelta

class random_dataframe:
    '''
    This class provides user with the flexibility to build a random dataframe if the user doesn't want to use actual data.
    It gives user flexibility to choose their seed, number of rows, channels, channel weights, lead rate, conversion rate
    
    '''
    
    def __init__(self):
        pass
    
    def create_random_df(self,
                         seed=None,
                         nrows=10000,
                         channels = ['FB','X','Email','Direct','Google'], 
                         channel_weights = [0.2]*5,
                         lead_rate = 0.3,
                         conversion_rate = 0.1):
        
        np.random.seed(seed) # Set your seed to have repeatability. 
        
        #Code to generate random values
        user_id = np.random.choice(2000,nrows)
        chan_list = np.random.choice(len(channels),nrows,p=channel_weights)
        lead = [0,1]
        lead_weights = [1-lead_rate,lead_rate]
        leads = np.random.choice(2,nrows,p=lead_weights)
        conv = [0,1]
        conv_weights = [1-conversion_rate,conversion_rate]
        conversions = np.random.choice(2,nrows,p=conv_weights)
        today = datetime.today()
        delta_dates = list(np.random.choice(30,nrows))
        dates_list = [today + timedelta(days = -int(x)) for x in delta_dates]
        df = pd.DataFrame({'user_id':user_id,
                          'channel_id':chan_list,
                          'event_date':dates_list,
                          'lead_generation': leads,
                          'conversion':conversions})
        chan_df = pd.DataFrame({'channel':channels,'channel_id':list(range(len(channels)))})
        df = pd.merge(df,chan_df,on = 'channel_id',how = 'inner')
        df = df[['user_id','channel','event_date','lead_generation','conversion']]
        time_df = pd.DataFrame(df.groupby(['user_id','event_date']).size()).reset_index()
        df = pd.merge(df,time_df,on = ['user_id','event_date'],how = 'left')
        
        # This piece of code below ensures that there are no duplicates in event_date per user to mimic real life users. 
        time_df = df[df[0]>1].drop(columns = 0)
        df = df[df[0]==1].drop(columns = 0)
        time_df = time_df.sample(frac=1).drop_duplicates(subset='user_id')
        df = pd.concat([df,time_df])
        df = df.sort_values(by = ['user_id','event_date']).reset_index().drop(columns = ['index'])
        
        #This piece of code below ensures that there is only 1 conversion per user. 
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
        
        #This piece of code ensures that there is at least one lead generation event on or before conversion.
        
        conv_df = df[df['conversion']==1][['user_id','event_date']].rename(columns = {'event_date':'conv_date'})
        conv_df = pd.merge(df,conv_df,on = 'user_id',how = 'inner')
        conv_df['delta'] = (conv_df['conv_date'] - conv_df['event_date']).dt.days
        conv_df = conv_df[conv_df['delta']>=0].drop(columns =['conv_date','delta'])
        user_df = pd.DataFrame(conv_df.groupby(['user_id'])['lead_generation'].sum()).reset_index().rename(columns = {'lead_generation':'count'})
        conv_df = pd.merge(conv_df,user_df,on='user_id',how = 'left')
        conv_df = conv_df[conv_df['count']==0].sample(frac=1).drop_duplicates(subset='user_id')
        df = pd.merge(df,conv_df[['user_id','event_date']],on = ['user_id','event_date'],how = 'left', indicator = True)
        df['lead_generation'] = np.where(df['_merge']=='both',1,df['lead_generation'])
        df = df.drop(columns = ['_merge'])
        return df
    
# random_dataframe = random_dataframe()

class attribution_subset:
    def __init__(self):
        pass
    
    def get_data_in_attr_window(self,
                                dataframe = pd.DataFrame({}),
                                attribution_window = 30,
                                path_transforms = 'unique'
                               ):
        
        filter_df = dataframe[dataframe['conversion']==1][['user_id','event_date']].rename(columns = {'event_date':'conv_date'})
        dataframe = pd.merge(dataframe,filter_df,on = 'user_id',how = 'outer')
        dataframe['conv_date'] = dataframe['conv_date'].fillna(dataframe['event_date'])
        dataframe['delta'] = (dataframe['conv_date'] - dataframe['event_date']).dt.days
        dataframe = dataframe[((dataframe['delta']<=attribution_window) & 
                              (dataframe['delta'] >=0))].drop(columns =['conv_date','delta'])
        return dataframe
    
# attribution_df = attribution_df()
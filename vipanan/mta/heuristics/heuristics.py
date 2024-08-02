import pandas as pd
import numpy as np

class heuristics:
    def __init__(self):
        pass
    
    def get_first_touch_attribution(self,
                                    dataframe = pd.DataFrame({})
                                   ):
        conv_df = dataframe[dataframe['conversion']==1][['user_id']]
        dataframe = pd.merge(dataframe,conv_df,on = 'user_id',how = 'inner')
        ndf = dataframe.groupby(['user_id'])['event_date'].min().reset_index()
        dataframe = pd.merge(dataframe,ndf,on =['user_id','event_date'],how = 'inner')
        ndf = pd.DataFrame(dataframe['user_id'].value_counts()).reset_index().rename(columns ={'user_id':'count',
                                                                                               'index':'user_id'})
        ndf = ndf[ndf['count']>1]
        if len(ndf) > 0:
            dataframe = pd.merge(dataframe,ndf,on = 'user_id',how = 'outer',indicator = True)
            ndf = dataframe[dataframe['_merge']=='both']
            ndf = ndf.sample(frac=1).drop_duplicates(subset='user_id')
            dataframe = dataframe[dataframe['_merge']=='left_only']
            dataframe = pd.concat([dataframe,ndf])
        ndf = pd.DataFrame(dataframe['channel'].value_counts()).reset_index().rename(columns ={'channel':'conversions',
                                                                                               'index':'channel'})
        return ndf
    
    def get_last_touch_attribution(self,
                                   dataframe = pd.DataFrame({}),
                                   channel = None
                                  ):
        conv_df = dataframe[dataframe['conversion']==1]
        
        if channel:
            mdf = conv_df[conv_df['channel']==channel]
            conv_df = conv_df[conv_df['channel']!=channel]
            mdf = pd.merge(dataframe,mdf[['user_id']], on = 'user_id',how = 'inner')
            mdf1 = mdf[mdf['channel']!= channel]
            
            mdf2 = pd.merge(mdf,mdf1[['user_id']],on = 'user_id',how = 'left',indicator= True)
            mdf2 = mdf2[mdf2['_merge']=='left_only'].drop(columns =['_merge'])
            mdf2 = mdf2[mdf2['conversion']==1]
            conv_df = pd.concat([conv_df,mdf2])
            
            mdf1 = mdf1.groupby(['user_id'])['event_date'].max().reset_index()
            mdf1 = pd.merge(mdf, mdf1, on = ['user_id','event_date'], how = 'inner')
            ndf = pd.DataFrame(mdf1['user_id'].value_counts()).reset_index().rename(columns ={'user_id':'count',
                                                                                               'index':'user_id'})
            ndf = ndf[ndf['count']>1]
            if len(ndf) > 0:
                mdf1 = pd.merge(mdf1,ndf,on = 'user_id',how = 'outer',indicator = True)
                ndf = mdf1[mdf1['_merge']=='both']
                ndf = ndf.sample(frac=1).drop_duplicates(subset='user_id')
                mdf1 = mdf1[mdf1['_merge']=='left_only']
                mdf1 = pd.concat([mdf1,ndf])
                mdf1 = mdf1.drop(columns = ['_merge','count'])

            conv_df = pd.concat([conv_df,mdf1])
        ndf = conv_df['channel'].value_counts().reset_index().rename(columns ={'channel':'conversions',
                                                                                               'index':'channel'})
        return ndf
    
heuristics = heuristics()


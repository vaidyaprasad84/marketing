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
        dataframe = pd.merge(dataframe,ndf,on = 'user_id',how = 'outer',indicator = True)
        ndf = dataframe[dataframe['_merge']=='both']
        ndf = ndf.sample(frac=1).drop_duplicates(subset='user_id')
        dataframe = dataframe[dataframe['_merge']=='left_only']
        dataframe = pd.concat([dataframe,ndf])
        ndf = pd.DataFrame(dataframe['channel'].value_counts()).reset_index().rename(columns ={'channel':'conversions',
                                                                                               'index':'channel'})
        return ndf
    
heuristics = heuristics()


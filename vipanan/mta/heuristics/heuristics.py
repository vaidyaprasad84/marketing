import pandas as pd
import numpy as np
from scipy.optimize import fsolve

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
    
    
    def get_time_decay_attribution(self,
                                   dataframe = pd.DataFrame({}),
                                   decay_type='exponential',
                                   A = 1):
        '''
        Here a default value of A = 1 is added. The user has an option to of course change the A.
        But changing A can lead to different solution and A = 1 is recommended. 
        Please refer to documentation for more details. 
        
        There are 3 options for decay time. 
        1. Exponential --> The last channel has maximum attribution and the attribution decays exponentially as go from last to first
        2. Linear --> The last channel has maximum attribution and the attribution decays linearly as go from last to first
        3. None --> There is no time decay. Meaning all channels have equal attribution. This is same as linear MTA model.
        '''
        
        conv_df = dataframe[dataframe['conversion']==1][['user_id']]
        conv_df = pd.merge(dataframe,conv_df,on = 'user_id', how = 'inner')
        conv_df1 = conv_df.groupby(['user_id']).size().reset_index().rename(columns = {0:'count'})
        conv_attr = []
        
        if decay_type == 'exponential': # Exponenial decay model
            d = dict({1:[1]}) # When there is just one channel, it will get 100% attribution
            n = 1 # This is just a dummy value added to ensure the next function runs. Its like a default value of n. 
            def func(x):
                return (1-np.exp((n+1)*x[0]))/(1-np.exp(x[0])) - (1+A)/A
            
            for i in range(len(conv_df1)):
                n = conv_df1.iloc[i]['count']
                if n not in d: # the look up is applied to minize computation
                    k = fsolve(func,[1])[0]
                    wt = []
                    for j in range(1,n+1):
                        wt.append(round(A*np.exp(k*j),3))
                    wt.reverse() # A reverse function has to be applied. Because the fn gives value from last channel to first while input data is sored from first to last.
                    d[n] = wt
                conv_attr += d[n]
                
        elif decay_type == 'linear': # Linear decay model
            ans = 1
        elif decay_type == None: #Linear MTA
            for i in range(len(conv_df1)):
                n = conv_df1.iloc[i]['count']
                conv_attr += [round(1/n,3)]*n
            
        conv_df['conv_attr'] = conv_attr
        ans = pd.DataFrame(conv_df.groupby(['channel'])['conv_attr'].sum()).reset_index()   
        return ans
                                 
heuristics = heuristics()


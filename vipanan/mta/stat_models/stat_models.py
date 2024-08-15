import pandas as pd
import numpy as np

class markov_chain_attribution:
    
    def __init__(self):
        pass
    
    def get_transition_matrix (self, 
                               dataframe = pd.DataFrame()
                              ):
        '''
        
        We assume the conversion paths to follow markov chain. 
        This function is used to create probability transition matrix. 
        
        '''
        dataframe = dataframe.set_index("user_id")
        # lagged df is created to get next transition state (next channel in this case) to build the transition matrix.
        lagged = dataframe.groupby(level="user_id").shift(-1)[['channel']].rename(columns = {'channel':'nxt_channel'})
        dataframe = pd.concat([dataframe,lagged],axis = 1)
        # Both conversion and non-conversions are assumed a terminal states. 
        dataframe['nxt_channel'] =  dataframe['nxt_channel'].fillna(dataframe['conversion'])
        dataframe = dataframe.reset_index()
        pivot_df = pd.pivot_table(dataframe,values = 'user_id', index = 'channel',columns = ['nxt_channel'], aggfunc = 'count')
        start_df = pd.DataFrame(dataframe.groupby(['user_id']).first()['channel'].value_counts()).rename(columns={'channel':'Start'}).T
        pivot_df = pd.concat([start_df,pivot_df]).fillna(0).rename(columns = {0:'Non_Conversion',1:'Conversion'})
        pivot_df.loc[:,pivot_df.columns]=pivot_df.div(pivot_df.sum(1),0)
        cols = pivot_df.columns.to_list()
        pivot_df = pd.concat([pivot_df,pd.DataFrame(np.zeros((2,len(cols))),columns = cols).set_axis(['Conversion',
                                                                                                      'Non_Conversion'])])
        pivot_df.loc['Conversion','Conversion'] = 1 # Terminal State. So probability to stay in the same state is 1. 
        pivot_df.loc['Non_Conversion','Non_Conversion'] = 1 # Terminal State. So probability to stay in the same state is 1. 
        pivot_df['Start'] = 0.000 # There is no way that user can go back to start state. Its just a mathematical construct. User can never go to start state once journey beguns.
        pivot_df = round(pivot_df,3)
        cols = pivot_df.index.values.tolist()
        pivot_df = pivot_df[cols]
        return pivot_df
    
    def get_removal_effects (self, 
                             dataframe = pd.DataFrame({})
                            ):
        conv_df = dataframe[dataframe['conversion']==1]
        total_conv = conv_df['conversion'].sum()
        conv_df = conv_df[['user_id']]
        conv_df = pd.merge(dataframe,conv_df,on = 'user_id', how = 'inner')
        channel_list = dataframe['channel'].value_counts().index.tolist()
        
        removal_effect = []
        
        for channel in channel_list:
            removal_effect.append(conv_df[conv_df['channel']==channel]['user_id'].nunique())
        
        norm_removal_effect = np.array(removal_effect)
        norm_removal_effect = norm_removal_effect/sum(norm_removal_effect)
        removal_effect = [x/total_conv for x in removal_effect]
        
        ans = pd.DataFrame({'channel':channel_list, 'Removal Effect':removal_effect, 
                            'Normalized Removal Effect':norm_removal_effect})
        
        return ans
    
    def get_markov_attribution (self, 
                                dataframe=pd.DataFrame({})
                               ):
        conv_df = dataframe[dataframe['conversion']==1]
        total_conv = conv_df['conversion'].sum()
        removal_effect = self.get_removal_effects(dataframe)
        removal_effect['conv_attr'] = removal_effect['Normalized Removal Effect']*total_conv
        removal_effect = removal_effect[['channel','conv_attr']]
        return removal_effect


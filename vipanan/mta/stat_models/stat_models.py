import pandas as pd
import numpy as np

class markov_chain_attribution:
    
    def __init__(self):
        pass
    
    def get_transition_matrix (self, 
                               dataframe = pd.DataFrame()
                              ):
        channel_list = dataframe['channel'].value_counts().index.tolist()
        num_channels = len(channel_list)
        tm = np.zeros((num_channels+1,num_channels+2))
        user_list = dataframe['user_id'].value_counts().index.tolist()
#         user_list.sort()
        
        for i in range(len(user_list)):
            sub_df = dataframe[dataframe['user_id']==user_list[i]].reset_index()
            n = len(sub_df)
            for j in range(n):
                cur_channel = sub_df.iloc[j]['channel']
                cur_index = channel_list.index(cur_channel) 
                if j == 0:
                    tm[j,cur_index] += 1
                else:
                    prev_channel = sub_df.iloc[j-1]['channel']
                    prev_index = channel_list.index(prev_channel)
                    tm[prev_index+1,cur_index] += 1
                if n == 1 or j == n-1:    
                    conv = sub_df.iloc[j]['conversion']
                    if conv == 1:
                        tm[cur_index+1,num_channels] += 1
                    else:
                        tm[cur_index+1,num_channels+1] += 1   
        col_names = channel_list + ['Conversion','Non_Conversion']
        tm = pd.DataFrame(tm,columns = col_names)
        row_index = ['Start'] + channel_list
        tm = tm.set_index(keys = [row_index])
        return tm
    
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
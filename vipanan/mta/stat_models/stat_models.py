import pandas as pd
import numpy as np

class markov_chain_attribution:
    
    def __init__(self):
        pass
    
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
        
        
markov_chain_attribution = markov_chain_attribution()


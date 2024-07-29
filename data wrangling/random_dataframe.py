import pandas as pd
import numpy as np
from datetime import datetime,timedelta

def random_df(seed=None,
              nrows=10000,
              channels = ['FB','X','Email','Direct','Google'], 
              channel_weights = [0.2]*5,
              conversion_rate = 0.1,
              start_date = None,
              delta = None):
    np.random.seed(seed)
    
    user_id = np.random.choice(200,nrows)
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
    return df


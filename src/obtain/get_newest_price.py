#!/usr/bin/env python
# coding: utf-8

# In[18]:


# get data
from binance.spot import Spot 
from constants import price_cols, PERIOD, SYMBOL, DURATION_EACH_DAY, DATA_POINT_ONE_DAY, NUM_OF_DAYS

# train data
import pandas as pd
from datetime import datetime

# enviroment
import os
from dotenv import load_dotenv
load_dotenv("../../env/app.env")


# In[13]:


BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
client = Spot(key=BINANCE_API_KEY, secret=BINANCE_API_SECRET)


# ## Read old prices

# In[30]:


price = pd.read_csv(f"../../datastore/price/{SYMBOL}_{PERIOD}.csv")
price


# ## Fetch new prices

# In[31]:


start_timestamp = price.iloc[-1, 0]
today = datetime.utcnow().replace(hour=23, minute=0, second=0, microsecond=0)
today_timestamp = int(today.timestamp() * 1000)


# In[32]:


for timestamp in range(start_timestamp, today_timestamp, DURATION_EACH_DAY):
    data = client.klines(SYMBOL, PERIOD, limit=1000, startTime=timestamp, endTime=timestamp + DURATION_EACH_DAY)
    if start_timestamp == timestamp:
        new_price_df = pd.DataFrame(data, columns=price_cols)
    else:
        new_price_df = pd.concat([new_price_df, pd.DataFrame(data, columns=price_cols)], axis=0)

new_price_df.drop("Unused field, ignore", axis=1, inplace=True)


# ## Concatenate old and new prices and save

# In[34]:


price = pd.concat([price, new_price_df])
price.drop_duplicates("Kline open time", inplace=True)
price.to_csv(f"../../datastore/price/{SYMBOL}_{PERIOD}.csv", index=False)


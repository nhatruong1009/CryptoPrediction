
price_cols= [
        'Kline open time',   #if a range of time
        'Open price',       #First trade price in the time range
        'High price',       #max buy in time
        'Low price',        #Min sell in time
        'Close price',      #Last trade price in the time range
        'Volume',           #Sum in time
        'Kline Close time',
        'Quote asset volume',   #Sum in time
        'Number of trades',     #Sum in time
        'Taker buy base asset volume',  #Sum in time
        'Taker buy quote asset volume', #Sum in time
        'Unused field, ignore'  # not use 
]

PERIOD = "15m"

SYMBOL = "BTCUSDT"

DURATION_EACH_DAY = 60 * 60 * 24 * 1000

DATA_POINT_ONE_DAY = DURATION_EACH_DAY // (15 * 60 * 1000)

NUM_OF_DAYS = 1800
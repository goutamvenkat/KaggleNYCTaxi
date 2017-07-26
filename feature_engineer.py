import pandas as pd
from geopy.distance import great_circle
from dateutil.parser import parse
from geopy.geocoders import Nominatim
import time
from multiprocessing import cpu_count, Pool
import numpy as np


def get_distance(lat1, long1, lat2, long2):
    return great_circle((lat1, long1), (lat2, long2)).miles
def get_borough(lat, lng, retries=3):
    if retries <= -1: return None
    try:
        geolocator = Nominatim()
        location = geolocator.reverse('{}, {}'.format(lat, lng))
        return location.address.split(', ')[2]
    except Exception as e:
        print 'Too many requests. Waiting for 2 mins for {}, {}'.format(lat, lng)
        time.sleep(30)
        return get_borough(lat, lng, retries-1)
        


def do_stuff(df):
    df['distance'] = df.apply(lambda row : get_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], 
                                                        row['dropoff_longitude']), axis=1)
    df['month'] = df.apply(lambda row: parse(row['pickup_datetime']).month, axis=1)
    df['day'] = df.apply(lambda row: parse(row['pickup_datetime']).weekday(), axis=1)
    df['pickup_hour'] = df.apply(lambda row: parse(row['pickup_datetime']).hour, axis=1)
    df['dropoff_hour'] = df.apply(lambda row: parse(row['dropoff_datetime']).hour, axis=1)
    df['pickup_borough'] = df.apply(lambda row: get_borough(row['pickup_latitude'], row['pickup_longitude']), axis=1)
    return df

# def parallelize_dataframe(df, func):
#     df_split = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df

if __name__ == '__main__':
    print 'Reading CSV'
    dataframe = pd.read_csv('train/train.csv', index_col=0)
    num_cores = cpu_count() #Number of CPU cores on your system
    num_partitions = num_cores #Define as many partitions as you want
    print 'Spliting dataframe'
    df_split = np.array_split(dataframe, num_partitions)
    print 'Instantiating a pool of workers'
    pool = Pool(num_cores)
    df = pd.concat(pool.map(do_stuff, df_split))
    pool.close()
    pool.join()
    print df.head()
# Import libraries
import glob
import pandas as pd
import numpy as np
import math

# Get CSV files list from a folder
path = 'C:\\Users\loran\Desktop\Data_to_match_epochvarjo\match_epoch_joanvarjo'
csv_files_joan = glob.glob(path + "/*.csv")

data_varjo_vehicle1 = pd.read_csv(
    'C:\\Users\loran\Desktop\Data_to_match_epochvarjo\Varjo_data\Varjo_vehicle1.csv_2022-08-18 143546.csv',
    sep=',')

data_varjo_vehicle2 = pd.read_csv(
    'C:\\Users\loran\Desktop\Data_to_match_epochvarjo\Varjo_data\Varjo_vehicle2.csv_2022-08-18 143551.csv',
    sep=',')


# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file, sep=';') for file in csv_files_joan)

# Concatenate all DataFrames
df_all_joan_trails = pd.concat(df_list, ignore_index=True)
df_all_joan_trails.values.tolist()

df_all_joan_trails['date'] = pd.to_datetime(df_all_joan_trails['Carla Interface.time'])
df_all_joan_trails['date'] = df_all_joan_trails['date'].round('ms')
data_varjo_vehicle1['date'] = pd.to_datetime(data_varjo_vehicle1['epoch']*1000)
data_varjo_vehicle1['date'] = data_varjo_vehicle1['date'].round('ms')

print(df_all_joan_trails['date'])
print(data_varjo_vehicle1['date'])

merged_varjo_joan = pd.merge_asof(df_all_joan_trails['date'],
                                  data_varjo_vehicle1['date'],
                                  left_index=True,
                                  right_index=True,
                                  direction='nearest',
                                  tolerance=1)

# saving the dataframe
merged_varjo_joan.to_csv('GFG.csv')

# list1 = list(round(df_all_joan_trails['Carla Interface.time'][1:]/1000))
# list2 = list(data_varjo_vehicle1['epoch'])
# # c = list(data_varjo_vehicle2['epoch'])
# print('joandata:', len(list1))
# print('varjodata:',len(list2))
#
#
# nearest_epoch_varjo = list(map(lambda y:min(list2, key=lambda x:abs(x-y)),list1))
#
# nearest_epoch_varjo_filtered = [item for item in list2 if item in nearest_epoch_varjo]
# # print(len(nearest_epoch_varjo_filtered))
# # print(nearest_epoch_varjo_filtered)
# # #39% error now!!!
#
# index_varjo_pandas = []
# for i in range(len(nearest_epoch_varjo_filtered)):
#     index = list2.index(nearest_epoch_varjo_filtered[i])
#     index_varjo_pandas.append(index)
#
# index_varjo_pandas = index_varjo_pandas
# index_epoch_data = data_varjo_vehicle1['epoch'][index_varjo_pandas]
# index_hmd_data = data_varjo_vehicle1['HMD_rotation'][index_varjo_pandas]
# index_gaze_data = data_varjo_vehicle1['gaze_forward'][index_varjo_pandas]
#
# for i in range(len(nearest_epoch_varjo)):
#     if nearest_epoch_varjo[i] == index_epoch_data[i]:


# print(len(index_varjo_pandas))
# print(len(nearest_epoch_varjo_filtered))

#fill up missing data
# for i in range(len(nearest_epoch_varjo)):
#     if nearest_epoch_varjo[i] ==
# print(data_varjo_vehicle1['HMD_rotation'][index_varjo_pandas])

# df_all_joan_trails['HMD_rotation'] = data_varjo_vehicle1['HMD_rotation'][index_varjo_pandas]
# print(df_all_joan_trails)

# dictionary of lists

# # saving the dataframe
# merged_varjo_joan.to_csv('GFG.csv')
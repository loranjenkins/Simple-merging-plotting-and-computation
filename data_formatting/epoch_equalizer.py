# Import libraries
from pathlib import Path
import pandas as pd


#locate all files joan
files_directory= 'C:\\Users\localadmin\Desktop\ExperimentOlgerArkady'

#-----vehicle 1 data -------#
dataset1_varjo_vehicle1 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\ExperimentOlgerArkady\Varjo\Varjo_data_vehicle1_2022-09-15 142838.csv',
    sep=',')
dataset2_varjo_vehicle1 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\ExperimentOlgerArkady\Varjo\Varjo_data_vehicle1_2022-09-15 142838.csv',
    sep=',')

data_varjo_vehicle1 = pd.concat([dataset1_varjo_vehicle1, dataset2_varjo_vehicle1], ignore_index=True)
print(dataset1_varjo_vehicle1)
print(dataset2_varjo_vehicle1)
print(data_varjo_vehicle1)

#-----vehicle 2 data -------#
dataset1_varjo_vehicle2 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\ExperimentOlgerArkady\Varjo\Varjo_data_vehicle2_2022-09-15 142842.csv',
    sep=',')
dataset2_varjo_vehicle2 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\ExperimentOlgerArkady\Varjo\Varjo_data_vehicle2_2022-09-15 142842.csv',
    sep=',')

data_varjo_vehicle2 = pd.concat([dataset1_varjo_vehicle1, dataset2_varjo_vehicle1], ignore_index=True)


# data_varjo_vehicle1['date'] = pd.to_datetime(data_varjo_vehicle1['epoch_vehicle1'])
# data_varjo_vehicle2['date'] = pd.to_datetime(data_varjo_vehicle2['epoch_vehicle2'])
#
# #Combining Varjo data to each trail seperate
#
# trail = 0
#
# for file in Path(files_directory).glob('*.csv'):
#     df = pd.read_csv(file, sep=';')
#     df.drop(df.loc[df['Carla Interface.time'] == 0].index, inplace=True)
#     df = df.drop_duplicates(subset=['Carla Interface.time'], keep=False)
#     df['date'] = pd.to_datetime(df['Carla Interface.time'])
#
#     merged_varjo_joan_vehicle1 = pd.merge_asof(df.sort_values('date'),
#                                       data_varjo_vehicle1.sort_values('date'),
#                                       on='date',
#                                       direction='nearest',
#                                       )
#
#     merged_varjo_joan_total = pd.merge_asof(merged_varjo_joan_vehicle1.sort_values('date'),
#                                                data_varjo_vehicle2.sort_values('date'),
#                                                on='date',
#                                                direction='nearest',
#                                                )
#
#     files_directory_combined="C:\\Users\localadmin\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined\Joan_Varjo_combinded_Trail#"
#     merged_varjo_joan_total.to_csv('{}{}.csv'.format(files_directory_combined, str(trail)), index=False)
#     trail += 1




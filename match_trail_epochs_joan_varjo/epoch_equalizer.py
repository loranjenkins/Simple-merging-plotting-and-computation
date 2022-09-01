# Import libraries
from pathlib import Path
import pandas as pd


#locate all files joan
files_directory= 'C:\\Users\localadmin\Desktop\Match_epochs\JoanMultitrail'

#locate csv files varjo
data_varjo_vehicle1 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\Match_epochs\Varjo\Varjo_experiment_data.csv_2022-09-01 111840.csv',
    sep=',')

data_varjo_vehicle2 = pd.read_csv(
    'C:\\Users\localadmin\Desktop\Match_epochs\Varjo\Varjo_experiment_data.csv_2022-09-01 111840.csv',
    sep=',')

data_varjo_vehicle1['date'] = pd.to_datetime(data_varjo_vehicle1['epoch'])
data_varjo_vehicle2['date'] = pd.to_datetime(data_varjo_vehicle2['epoch'])

#Combining Varjo data to each trail seperate

trail = 0

for file in Path(files_directory).glob('*.csv'):
    df = pd.read_csv(file, sep=';')
    df.drop(df.loc[df['Carla Interface.time'] == 0].index, inplace=True)
    df = df.drop_duplicates(subset=['Carla Interface.time'], keep=False)
    df['date'] = pd.to_datetime(df['Carla Interface.time'])

    merged_varjo_joan_vehicle1 = pd.merge_asof(df.sort_values('date'),
                                      data_varjo_vehicle1.sort_values('date'),
                                      on='date',
                                      direction='nearest',
                                      )

    merged_varjo_joan_total = pd.merge_asof(merged_varjo_joan_vehicle1.sort_values('date'),
                                               data_varjo_vehicle2.sort_values('date'),
                                               on='date',
                                               direction='nearest',
                                               )

    files_directory_combined="C:\\Users\localadmin\Desktop\Match_epochs\JoancombinedVarjo\Joan_Varjo_combinded_Trail#"
    merged_varjo_joan_total.to_csv('{}{}.csv'.format(files_directory_combined, str(trail)))
    trail += 1




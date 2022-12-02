import pandas as pd
import os

path_to_data_csv = os.path.join('..', 'data_folder', 'crt_hitter.csv')
global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

print(global_crt_index.mean())
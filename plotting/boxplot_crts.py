import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_all_conditions.csv')
path_to_data_csv_index = os.path.join('..', 'data_folder', 'crt_index_all_conditions.csv')

global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt.replace(0, np.nan, inplace=True)
global_crt_index = pd.read_csv(path_to_data_csv_index, sep=',')

medians_crt = global_crt.median()
medians_index = global_crt_index.median()

path_to_saved_dict = os.path.join('..', 'data_folder', 'medians_crt.csv')
path_to_saved_dict = os.path.join('..', 'data_folder', 'medians_crt_index.csv')

df1 = pd.DataFrame({'median_50_50': [medians_crt[0]]})
df2 = pd.DataFrame({'median_55_45': [medians_crt[1]]})
df3 = pd.DataFrame({'median_60_40': [medians_crt[2]]})
pd.concat([df1, df2, df3], axis=1).to_csv(path_to_saved_dict, index=False)

df3 = pd.DataFrame({'median_50_50': [round(medians_index[0])]})
df4 = pd.DataFrame({'median_55_45': [round(medians_index[1])]})
df5 = pd.DataFrame({'median_60_40': [round(medians_index[2])]})
pd.concat([df3, df4, df5], axis=1).to_csv(path_to_saved_dict, index=False)


a = sns.boxplot(data=global_crt)

# for _, line_list in a.items():
#     for line in line_list:
#         line.set_color('r')

plt.ylabel('CRT [s]')
plt.xlabel('Condition')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

# path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_first_vehicle_exit.csv')
path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_at_average_exit.csv')

# path_to_data_csv_index = os.path.join('..', 'data_folder', 'crt_index_all_conditions.csv')

# path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_last_vehicle_exit.csv')
# path_to_data_csv_index = os.path.join('..', 'data_folder', 'crt_index_all_conditions.csv')

global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt.replace(0, np.nan, inplace=True)
# print(global_crt['Condition 1'].mean())
# global_crt_index = pd.read_csv(path_to_data_csv_index, sep=',')
# columns_titles = ["Condition 1", "Condition 2", "Condition 3"]
# global_crt = global_crt.reindex(columns=columns_titles)

medians_crt = global_crt.median()
# medians_index = global_crt_index.median()

path_to_saved_dict1 = os.path.join('..', 'data_folder', 'medians_crt.csv')
# path_to_saved_dict = os.path.join('..', 'data_folder', 'medians_crt_index.csv')

df1 = pd.DataFrame({'median_60_40': [round(medians_crt[0], 2)]})
df2 = pd.DataFrame({'median_50_50': [round(medians_crt[1], 2)]})
df3 = pd.DataFrame({'median_55_45': [round(medians_crt[2], 2)]})
pd.concat([df1, df2, df3], axis=1).to_csv(path_to_saved_dict1, index=False)

# df4 = pd.DataFrame({'median_60_40': [medians_index[2]]})
# df5 = pd.DataFrame({'median_50_50': [medians_index[0]]})
# df6 = pd.DataFrame({'median_55_45': [medians_index[1]]})
# pd.concat([df4, df5, df6], axis=1).to_csv(path_to_saved_dict, index=False)

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

sns.boxplot(data=global_crt, **PROPS)

plt.title('Comparison CRT all conditions')
plt.ylabel('CRT [s]')
# plt.xlabel('Condition')
plt.show()


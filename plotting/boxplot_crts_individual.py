import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_who_is_ahead.csv')
path_to_data_csv_index = os.path.join('..', 'data_folder', 'crt_index_who_is_ahead.csv')

global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt.replace(0, np.nan, inplace=True)
global_crt_index = pd.read_csv(path_to_data_csv_index, sep=',')

medians_crt = global_crt.median()
medians_index = global_crt_index.median()

path_to_saved_dict1 = os.path.join('..', 'data_folder', 'medians_crt_who_is_ahead.csv')
path_to_saved_dict = os.path.join('..', 'data_folder', 'medians_crt_index_who_is_ahead.csv')

df1 = pd.DataFrame({'median_50_50': [medians_crt[0]]})
df2 = pd.DataFrame({'median_45_55_right': [medians_crt[1]]})
df3 = pd.DataFrame({'median_55_45_left': [medians_crt[2]]})
df4 = pd.DataFrame({'median_40_60_right': [medians_crt[3]]})
df5 = pd.DataFrame({'median_60_40_left': [medians_crt[4]]})
pd.concat([df1, df2, df3, df4, df5], axis=1).to_csv(path_to_saved_dict1, index=False)

df6 = pd.DataFrame({'median_index_50_50': [round(medians_index[0])]})
df7 = pd.DataFrame({'median_index_45_55_right': [round(medians_index[1])]})
df8 = pd.DataFrame({'median_index_55_45_left': [round(medians_index[2])]})
df9 = pd.DataFrame({'median_index_40_60_right': [round(medians_index[3])]})
df10 = pd.DataFrame({'median_index_60_40_left': [round(medians_index[4])]})
pd.concat([df6, df7, df8, df9, df10], axis=1).to_csv(path_to_saved_dict, index=False)

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

sns.boxplot(data=global_crt, **PROPS)

plt.title('Comparison CRT depending on headway')
plt.ylabel('CRT [s]')
plt.xlabel('Condition')
plt.show()


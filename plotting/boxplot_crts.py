import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
import pingouin as pg


path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_who_is_first_exit_interactive.csv')

global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt.replace(0, np.nan, inplace=True)

dict1 = {'condition': [], 'value': []}
dict2 = {'condition': [], 'value': []}
dict3 = {'condition': [], 'value': []}

individual_list_c1 = list(global_crt['Condition 1'])
individual_list_c2 = list(global_crt['Condition 2'])
individual_list_c3 = list(global_crt['Condition 3'])

for i in individual_list_c1:
    dict1['value'].append(i)
dict1['condition'] = ['condition1'] * len(dict1['value'])

for i in individual_list_c2:
    dict2['value'].append(i)

dict2['condition'] = ['condition2'] * len(dict2['value'])

for i in individual_list_c3:
    dict3['value'].append(i)

dict3['condition'] = ['condition3'] * len(dict3['value'])


df1 = pd.DataFrame.from_dict(dict1).dropna()

df2 = pd.DataFrame.from_dict(dict2).dropna()

df3 = pd.DataFrame.from_dict(dict3).dropna()

df = pd.concat([df1, df2, df3], ignore_index=True)


a = pg.anova(dv='value', between='condition', data=df, detailed=True)
b = pg.anova(dv='value', between='condition', data=df, detailed=False)
print(a)
print('-----')
print(b)
print('-----')


t1, p1 = ttest_ind(list(global_crt.iloc[:, 2].dropna()), list(global_crt.iloc[:, 1].dropna()))
print('Pearson values t and p: ', t1, p1)

medians_crt = global_crt.median()

# path_to_saved_dict1 = os.path.join('..', 'data_folder', 'medians_crt.csv')
#
#
# df1 = pd.DataFrame({'median_50_50': [round(medians_crt[0], 2)]})
# df2 = pd.DataFrame({'median_55_45': [round(medians_crt[1], 2)]})
# df3 = pd.DataFrame({'median_60_40': [round(medians_crt[2], 2)]})
#
# pd.concat([df1, df2, df3], axis=1).to_csv(path_to_saved_dict1, index=False)


PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

fig, ax1 = plt.subplots()
sns.boxplot(data=global_crt, **PROPS)

fig.suptitle('Comparison all CRTs between conditions')
ax1.set(ylabel='CRT [s]')
ax1.set_xticklabels(["Condition 1:\n 50-50 km/h", "Condition 2:\n 55-45 km/h", "Condition 3:\n 60-40 km/h"])
plt.show()


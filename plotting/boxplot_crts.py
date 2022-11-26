import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_who_is_first_exit.csv')
# path_to_data_csv_crt = os.path.join('..', 'data_folder', 'crt_at_average_exit.csv')


global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt.replace(0, np.nan, inplace=True)
columns_titles = ["Condition 1", "Condition 2", "Condition 3"]
global_crt = global_crt.reindex(columns=columns_titles)

medians_crt = global_crt.median()


path_to_saved_dict1 = os.path.join('..', 'data_folder', 'medians_crt.csv')


df1 = pd.DataFrame({'median_50_50': [round(medians_crt[0], 2)]})
df2 = pd.DataFrame({'median_55_45': [round(medians_crt[1], 2)]})
df3 = pd.DataFrame({'median_60_40': [round(medians_crt[2], 2)]})

pd.concat([df1, df2, df3], axis=1).to_csv(path_to_saved_dict1, index=False)


PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

fig, ax1 = plt.subplots()
sns.boxplot(data=global_crt, **PROPS)

fig.suptitle('Comparison CRT all conditions')
ax1.set(ylabel='CRT [s]')
# ax1.set_ylabel('CRT [s]')
ax1.set_xticklabels(["Condition 1:\n 50-50 km/h", "Condition 2:\n 55-45 km/h", "Condition 3:\n 60-40 km/h"])
plt.show()


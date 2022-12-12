import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind, ttest_ind_from_stats

path_to_data_csv_crt = os.path.join('..', 'data_folder', 'boxplot_who_is_ahead.csv')


global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
# global_crt.replace(0, np.nan, inplace=True)

global_crt_condition2 = global_crt.iloc[:, 0:2]
global_crt_condition3 = global_crt.iloc[:, 2:4].dropna()

t, p = ttest_ind(list(global_crt_condition2.iloc[:,0]), list(global_crt_condition2.iloc[:,1]))
t1, p1 = ttest_ind(list(global_crt_condition3.iloc[:,0]), list(global_crt_condition3.iloc[:,1]))


PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

fig, ax1 = plt.subplots()
sns.boxplot(data=global_crt_condition2, **PROPS)
fig.suptitle('Comparison average gaze behavior condition 2 (55-45 km/h)')
ax1.set(ylabel='Fixation on opponent [%]')
ax1.set_xticklabels(["Participant is ahead", "Participant is behind"])
ax1.plot([], [], ' ', label='t: ' + str(round(t, 3)))
ax1.plot([], [], ' ', label='p: ' + "{:.2e}".format(p))
ax1.legend(loc='best')

fig, ax2 = plt.subplots()
sns.boxplot(data=global_crt_condition3, **PROPS)

fig.suptitle('Comparison average gaze behavior condition 3 (60-40 km/h)')
ax2.set(ylabel='Fixation on opponent [%]')
ax2.set_xticklabels(["Participant is ahead", "Participant is behind"])
ax2.plot([], [], ' ', label='t: ' + str(round(t1, 3)))
ax2.plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))
ax2.legend(loc='best')

plt.show()


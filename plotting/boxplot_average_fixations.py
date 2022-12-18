import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from scipy.stats import ttest_ind
import pingouin as pg

path_to_data_csv_crt = os.path.join('..', 'data_folder', 'boxplot_who_is_ahead.csv')

global_crt = pd.read_csv(path_to_data_csv_crt, sep=',')
global_crt = global_crt[
    ['behind_fixations_55_45', 'ahead_fixations_55_45', 'behind_fixations_60_40', 'ahead_fixations_60_40']]

global_crt_condition2 = global_crt.iloc[:, 0:2]
global_crt_condition3 = global_crt.iloc[:, 2:4].dropna()

t, p = ttest_ind(list(global_crt_condition2.iloc[:, 1]), list(global_crt_condition2.iloc[:, 0]))
t1, p1 = ttest_ind(list(global_crt_condition3.iloc[:, 1]), list(global_crt_condition3.iloc[:, 0]))
pd.set_option('display.max_columns', None)

a = pg.ttest(list(global_crt_condition2.iloc[:, 1]), list(global_crt_condition2.iloc[:, 0]))
b = pg.ttest(list(global_crt_condition3.iloc[:, 1]), list(global_crt_condition3.iloc[:, 0]))
print(a)
print(b)

PROPS = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
    'medianprops': {'color': 'red'},
    'whiskerprops': {'color': 'black'},
    'capprops': {'color': 'black'}
}

fig, ax1 = plt.subplots()
sns.boxplot(data=global_crt_condition2, **PROPS)
fig.suptitle('Comparison average gaze behavior condition 2 (55-45 km/h)')
ax1.set(ylabel='Fixation on opponent [%]')
ax1.set_xticklabels(["Participant is behind", "Participant is ahead"])
ax1.plot([], [], ' ', label='t: ' + str(round(t, 3)))
ax1.plot([], [], ' ', label='p: ' + "{:.2e}".format(p))
ax1.legend(loc='best')

fig, ax2 = plt.subplots()
sns.boxplot(data=global_crt_condition3, **PROPS)

fig.suptitle('Comparison average gaze behavior condition 3 (60-40 km/h)')
ax2.set(ylabel='Fixation on opponent [%]')
ax2.set_xticklabels(["Participant is behind", "Participant is ahead"])
ax2.plot([], [], ' ', label='t: ' + str(round(t1, 3)))
ax2.plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))
ax2.legend(loc='best')

plt.show()

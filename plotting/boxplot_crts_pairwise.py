import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from numpy import median

path_to_data_csv_crt1 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment1.csv')
path_to_data_csv_crt2 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment2.csv')
path_to_data_csv_crt3 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment3.csv')
path_to_data_csv_crt4 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment4.csv')
path_to_data_csv_crt5 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment5.csv')
path_to_data_csv_crt6 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment6.csv')
path_to_data_csv_crt7 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_experiment7.csv')
path_to_data_csv_crt8 = os.path.join('..', 'data_folder', 'experiment_data', 'crt_vehicle1_combined.csv')

def average(pandas):
    median = pandas.median()
    average = sum(list(median)) / len(list(median))
    return average

crt_experiment1 = pd.read_csv(path_to_data_csv_crt1, sep=',')
crt_experiment1.replace(0, np.nan, inplace=True)
average1 = average(crt_experiment1)

crt_experiment2 = pd.read_csv(path_to_data_csv_crt2, sep=',')
crt_experiment2.replace(0, np.nan, inplace=True)
average2 = average(crt_experiment2)

crt_experiment3 = pd.read_csv(path_to_data_csv_crt3, sep=',')
crt_experiment3.replace(0, np.nan, inplace=True)
average3 = average(crt_experiment3)

crt_experiment4 = pd.read_csv(path_to_data_csv_crt4, sep=',')
crt_experiment4.replace(0, np.nan, inplace=True)
average4 = average(crt_experiment4)

crt_experiment5 = pd.read_csv(path_to_data_csv_crt5, sep=',')
crt_experiment5.replace(0, np.nan, inplace=True)
average5 = average(crt_experiment5)

crt_experiment6 = pd.read_csv(path_to_data_csv_crt6, sep=',')
crt_experiment6.replace(0, np.nan, inplace=True)
average6 = average(crt_experiment6)

crt_experiment7 = pd.read_csv(path_to_data_csv_crt7, sep=',')
crt_experiment7.replace(0, np.nan, inplace=True)
average7 = average(crt_experiment7)

crt_vehicle1 = pd.read_csv(path_to_data_csv_crt8, sep=',')
crt_vehicle1.replace(0, np.nan, inplace=True)
average8 = average(crt_vehicle1)

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

fig, axes = plt.subplots(4, 2)

fig.suptitle('Vehicle 1 CRT for each experiment')

axes[0][0].set_title('Experiment 1 with Average crt: ' + str(round(average1, 2)))
axes[1][0].set_title('Experiment 2 with Average crt: ' + str(round(average2, 2)))
axes[2][0].set_title('Experiment 3 with Average crt: ' + str(round(average3, 2)))
axes[3][0].set_title('Experiment 4 with Average crt: ' + str(round(average4, 2)))
axes[0][1].set_title('Experiment 5 with Average crt: ' + str(round(average5, 2)))
axes[1][1].set_title('Experiment 6 with Average crt: ' + str(round(average6, 2)))
axes[2][1].set_title('Experiment 7 with Average crt: ' + str(round(average7, 2)))
axes[3][1].set_title('Experiments combined with Average crt: ' + str(round(average8, 2)))


sns.boxplot(data=crt_experiment1, ax=axes[0][0], **PROPS)
sns.boxplot(data=crt_experiment2, ax=axes[1][0], **PROPS)
sns.boxplot(data=crt_experiment3, ax=axes[2][0], **PROPS)
sns.boxplot(data=crt_experiment4, ax=axes[3][0], **PROPS)
sns.boxplot(data=crt_experiment5, ax=axes[0][1], **PROPS)
sns.boxplot(data=crt_experiment6, ax=axes[1][1], **PROPS)
sns.boxplot(data=crt_experiment7, ax=axes[2][1], **PROPS)
sns.boxplot(data=crt_vehicle1, ax=axes[3][1], **PROPS)


plt.subplots_adjust(hspace = 0.5)
fig.text(0.06, 0.5, "CRT [s]", va='center', rotation='vertical')
fig.text(0.5, 0.06, "Velocity vehicle 1 [km/h]", ha="center", va="center")
# fig.delaxes(axes[3][1])
# plt.xlabel('Velocity vehicle 1 [km/h]')
plt.show()


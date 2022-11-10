import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import seaborn as sns
import os
import pickle
from natsort import natsorted

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack

files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50'
trails = []
for file in Path(files_directory).glob('*.csv'):
    # trail_condition = plot_trail(file)
    trails.append(file)
trails = natsorted(trails, key=str)

hmd_rots_v1 = []
hmd_rots_v2 = []
for i in range(len(trails)):
    data = pd.read_csv(trails[i], sep=',')
    a = sum(list(data['HMD_rotation_vehicle1']))
    b = len(list(data['HMD_rotation_vehicle1']))
    c = sum(list(data['HMD_rotation_vehicle2']))
    d = len(list(data['HMD_rotation_vehicle2']))
    average_v1 = a/b
    average_v2 = c/d
    hmd_rots_v1.append(average_v1)
    hmd_rots_v2.append(average_v2)


a = sum(hmd_rots_v1)
b = len(hmd_rots_v1)
c = sum(hmd_rots_v2)
d = len(hmd_rots_v2)
print(a/b)
print(c/d)

data = pd.read_csv(trails[50], sep=',')
sns.histplot(list(data['HMD_rotation_vehicle1']))
# sns.histplot(list(data['HMD_rotation_vehicle2']))

# plt.hist(list(data['HMD_rotation_vehicle1']), 30)
a = sum(list(data['HMD_rotation_vehicle1']))
b = len(list(data['HMD_rotation_vehicle1']))
c = sum(list(data['HMD_rotation_vehicle2']))
d = len(list(data['HMD_rotation_vehicle2']))

# plt.axvline(a/b, 0, 2300, color='r')
# plt.axvline(c/d, 0, 2300, color='b')

plt.show()

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

files_directory = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
files_directory1 = r'D:\Thesis_data_all_experiments\Conditions\condition_55_45'
files_directory2 = r'D:\Thesis_data_all_experiments\Conditions\condition_60_40'


trails = []
for file in Path(files_directory).glob('*.csv'):
    # trail_condition = plot_trail(file)
    trails.append(file)

for file in Path(files_directory1).glob('*.csv'):
    # trail_condition = plot_trail(file)
    trails.append(file)

for file in Path(files_directory2).glob('*.csv'):
    # trail_condition = plot_trail(file)
    trails.append(file)

# hmd_rots_v1 = []
# hmd_rots_v2 = []
# for i in range(len(trails)):
#     data = pd.read_csv(trails[i], sep=',')
#     a = sum(list(data['HMD_rotation_vehicle1']))
#     b = len(list(data['HMD_rotation_vehicle1']))
#     c = sum(list(data['HMD_rotation_vehicle2']))
#     d = len(list(data['HMD_rotation_vehicle2']))
#     average_v1 = a/b
#     average_v2 = c/d
#     hmd_rots_v1.append(average_v1)
#     hmd_rots_v2.append(average_v2)
#
#
# a = sum(hmd_rots_v1)
# b = len(hmd_rots_v1)
# c = sum(hmd_rots_v2)
# d = len(hmd_rots_v2)
# print(a/b)
# print(c/d)
def sum_list(l):
    sum = 0
    for x in l:
        sum += x
    return sum

# list all files in the directory, assuming this directory
# contains only files csv files that you need to save
hmd_rots = []
for i in trails:
    data = pd.read_csv(i, sep=',')
    v1 = []
    v2 = []
    for i in data['HMD_rotation_vehicle1']:
        if i > 0.6:
            v1.append(i)
    for i in data['HMD_rotation_vehicle2']:
        if i > 0.6:
            v2.append(i)
    new_list = v1+v2
hmd_rots.append(new_list)

resultList = []

# Traversing in till the length of the input list of lists
for m in range(len(hmd_rots)):

   # using nested for loop, traversing the inner lists
   for n in range (len(hmd_rots[m])):

      # Add each element to the result list
      resultList.append(hmd_rots[m][n])

sns.histplot(resultList)

plt.show()

#
# for index in Path(trails).glob('*.csv')
#     list = trails[index]
#     print(list)
#     for i in Path(list).glob('*.csv'):
#         data = pd.read_csv(i, sep=',')
#         vehicle1 = data['HMD_rotation_vehicle1']
#         print(vehicle1)
#         vehicle2 = data['HMD_rotation_vehicle2']
#         hmd_rots.append(vehicle1, vehicle2)
#
# print(hmd_rots)
# sns.histplot(list(hmd_rots))
# sns.histplot(list(data['HMD_rotation_vehicle2']))

# plt.hist(list(data['HMD_rotation_vehicle1']), 30)
# a = sum(list(data['HMD_rotation_vehicle1']))
# b = len(list(data['HMD_rotation_vehicle1']))
# c = sum(list(data['HMD_rotation_vehicle2']))
# d = len(list(data['HMD_rotation_vehicle2']))

# plt.axvline(a/b, 0, 2300, color='r')
# plt.axvline(c/d, 0, 2300, color='b')

plt.show()

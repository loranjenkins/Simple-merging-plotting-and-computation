import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

import seaborn as sns
from natsort import natsorted

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack


def vehicle_xy_coordinates(intcolumn, data_csv):
    list_x = []
    list_y = []
    for i in range(len(data_csv.iloc[:, intcolumn])):
        transform_vehicle1 = eval(data_csv.iloc[i, intcolumn])
        x_loc = transform_vehicle1[0]
        y_loc = transform_vehicle1[1]

        list_x.append(x_loc)
        # list_x = list(dict.fromkeys(list_x))
        list_y.append(y_loc)
        # list_y = list(dict.fromkeys(list_y))

    return list_x, list_y

def average_nested(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))


# if __name__ == '__main__':
def probability_calc(path_to_data_csv, left_or_right):

    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    # data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=120,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=150)

    track = SymmetricMergingTrack(simulation_constants)

    xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
    xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, data)

    xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
    xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

    xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
    xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)


    index_of_mergepoint_vehicle1 = min(range(len(xy_coordinates_vehicle1)),
                                       key=lambda i: abs(xy_coordinates_vehicle1[i][1] - track.merge_point[1]))
    index_of_mergepoint_vehicle2 = min(range(len(xy_coordinates_vehicle2)),
                                       key=lambda i: abs(xy_coordinates_vehicle2[i][1] - track.merge_point[1]))

    if left_or_right =='right':
        if index_of_mergepoint_vehicle1 < index_of_mergepoint_vehicle2:
            return 1
        else:
            return 0

    if left_or_right == 'left':
        if index_of_mergepoint_vehicle1 < index_of_mergepoint_vehicle2:
                return 1
        else:
                return 0

def compute_average(folder_with_csv, side):
    trails = []
    for i in range(len(folder_with_csv)):
        path_per_c = []
        for file in Path(folder_with_csv[i]).glob('*.csv'):
            path_per_c.append(file)
        trails.append(path_per_c)

    probability = []
    for i in range(len(trails)):
        single_folder = trails[i]
        inner_list = []
        for file in range(len(single_folder)):
            c_p = probability_calc(str(single_folder[file]), side)
            inner_list.append(c_p)
        probability.append(inner_list)

    average_per_condition = []
    for list in range(len(probability)):
        probability_p_c = probability[list]
        average = sum(probability_p_c) / len(probability_p_c)
        average_per_condition.append(average)

    return average_per_condition

if __name__ == '__main__':
    #experiment1
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment1'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment1'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment1'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment1'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment1'

    right = [c40_60, c45_55, c50_50]
    left = [c55_45, c60_40]

    average_right_experiment1 = compute_average(right, 'right')
    average_left_experiment1 = compute_average(left, 'left')
    average_experiment1 = average_right_experiment1 + average_left_experiment1

    #experiment2
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment2'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment2'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment2'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment2'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment2'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment2 = compute_average(right, 'right')
    average_left_experiment2 = compute_average(left, 'left')
    average_experiment2 = average_right_experiment2 + average_left_experiment2


    #experiment3
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment3'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment3'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment3'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment3'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment3'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment3 = compute_average(right, 'right')
    average_left_experiment3 = compute_average(left, 'left')
    average_experiment3 = average_right_experiment3 + average_left_experiment3

    #experiment4
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment4'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment4'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment4'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment4'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment4'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment4 = compute_average(right, 'right')
    average_left_experiment4 = compute_average(left, 'left')
    average_experiment4 = average_right_experiment4 + average_left_experiment4

    #experiment5
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment5'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment5'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment5'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment5'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment5'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment5 = compute_average(right, 'right')
    average_left_experiment5 = compute_average(left, 'left')
    average_experiment5 = average_right_experiment5 + average_left_experiment5

    #experiment6
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment6'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment6'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment6'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment6'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment6'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment6 = compute_average(right, 'right')
    average_left_experiment6 = compute_average(left, 'left')
    average_experiment6 = average_right_experiment6 + average_left_experiment6

    #experiment7
    c40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\right\experiment7'
    c45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\right\experiment7'
    c50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment7'
    c55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\left\experiment7'
    c60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\left\experiment7'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment7 = compute_average(right, 'right')
    average_left_experiment7 = compute_average(left, 'left')
    average_experiment7 = average_right_experiment7 + average_left_experiment7

    average_list = [average_experiment1, average_experiment2, average_experiment3, average_experiment4,
                    average_experiment5, average_experiment6, average_experiment7]
    average_list = np.array(average_list)
    mean = np.mean(average_list, axis=0)
    print(mean)

    for i in range(len(average_experiment1)):
        experiment1 = plt.scatter(i, average_experiment1[i], marker = "+", color = 'b')

    for i in range(len(average_experiment2)):
        experiment2 = plt.scatter(i, average_experiment2[i], marker = "x", color = 'g')

    for i in range(len(average_experiment3)):
        experiment3 = plt.scatter(i, average_experiment3[i], marker = "v", color = 'r')

    for i in range(len(average_experiment4)):
        experiment4 = plt.scatter(i, average_experiment4[i], marker = "^", color = 'c')

    for i in range(len(average_experiment5)):
        experiment5 = plt.scatter(i, average_experiment5[i], marker = "_", color = 'm')

    for i in range(len(average_experiment6)):
        experiment6 = plt.scatter(i, average_experiment6[i], marker = "|", color = 'y')

    for i in range(len(average_experiment7)):
        experiment7 = plt.scatter(i, average_experiment7[i], marker = "o", color = 'k')

    x = np.array([0, 1, 2, 3, 4])
    my_xticks = ['40_60', '45_55', '50_50', '55_45', '60_40']
    plt.xticks(x, my_xticks)
    plt.title('Probibility vehicle 1 merging first')
    plt.ylabel('Probibility')
    plt.xlabel('Condition')

    plt.legend((experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7),
               ('Experiment1', 'Experiment2', 'Experiment3',
                'Experiment4', 'Experiment5', 'Experiment6',
                'Experiment7'),
               scatterpoints=1,
               loc='lower left',
               ncol=1,
               fontsize=8)

    plt.show()


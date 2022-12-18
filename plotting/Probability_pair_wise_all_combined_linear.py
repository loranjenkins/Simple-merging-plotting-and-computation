import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import scipy.stats as stats


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
        list_y.append(y_loc)

    return list_x, list_y


# if __name__ == '__main__':
def probability_calc(path_to_data_csv, left_or_right):
    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    # data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=125,  # original = 118 -> check in unreal
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

    if left_or_right == 'right':
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
    print(probability)

    average_per_condition = []
    for list in range(len(probability)):
        probability_p_c = probability[list]
        average = sum(probability_p_c) / len(probability_p_c)
        average_per_condition.append(average)

    return average_per_condition


if __name__ == '__main__':
    # experiment1
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment1'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment1'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment1'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment1'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment1'

    right = [c40_60, c45_55, c50_50]
    left = [c55_45, c60_40]

    average_right_experiment1 = compute_average(right, 'right')
    average_left_experiment1 = compute_average(left, 'left')
    average_experiment1 = average_right_experiment1 + average_left_experiment1


    # experiment2
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment2'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment2'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment2'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment2'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment2'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment2 = compute_average(right, 'right')
    average_left_experiment2 = compute_average(left, 'left')
    average_experiment2 = average_right_experiment2 + average_left_experiment2

    #experiment3
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment3'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment3'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment3'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment3'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment3'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment3 = compute_average(right, 'right')
    average_left_experiment3 = compute_average(left, 'left')
    average_experiment3 = average_right_experiment3 + average_left_experiment3

    #experiment4
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment4'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment4'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment4'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment4'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment4'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment4 = compute_average(right, 'right')
    average_left_experiment4 = compute_average(left, 'left')
    average_experiment4 = average_right_experiment4 + average_left_experiment4

    #experiment5
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment5'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment5'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment5'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment5'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment5'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment5 = compute_average(right, 'right')
    average_left_experiment5 = compute_average(left, 'left')
    average_experiment5 = average_right_experiment5 + average_left_experiment5

    #experiment6
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment6'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment6'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment6'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment6'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment6'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment6 = compute_average(right, 'right')
    average_left_experiment6 = compute_average(left, 'left')
    average_experiment6 = average_right_experiment6 + average_left_experiment6

    #experiment7
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment7'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment7'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment7'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment7'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment7'

    left = [c55_45, c60_40]
    right = [c40_60, c45_55, c50_50]

    average_right_experiment7 = compute_average(right, 'right')
    average_left_experiment7 = compute_average(left, 'left')
    average_experiment7 = average_right_experiment7 + average_left_experiment7
    print(average_experiment7)
    _40value = [average_experiment1[0], average_experiment2[0], average_experiment3[0], average_experiment4[0], average_experiment5[0], average_experiment6[0], average_experiment7[0]]
    _45value = [average_experiment1[1], average_experiment2[1], average_experiment3[1], average_experiment4[1], average_experiment5[1], average_experiment6[1], average_experiment7[1]]
    _50value = [average_experiment1[2], average_experiment2[2], average_experiment3[2], average_experiment4[2], average_experiment5[2], average_experiment6[2], average_experiment7[2]]
    _55value = [average_experiment1[3], average_experiment2[3], average_experiment3[3], average_experiment4[3], average_experiment5[3], average_experiment6[3], average_experiment7[3]]
    _60value = [average_experiment1[4], average_experiment2[4], average_experiment3[4], average_experiment4[4], average_experiment5[4], average_experiment6[4], average_experiment7[4]]

    dict = {'x_value': [], 'y_value': []}
    for i in range(7):
        dict['x_value'].append(40)
        dict['y_value'].append(_40value[i])

    for i in range(7):
        dict['x_value'].append(45)
        dict['y_value'].append(_45value[i])

    for i in range(7):
        dict['x_value'].append(50)
        dict['y_value'].append(_50value[i])

    for i in range(7):
        dict['x_value'].append(55)
        dict['y_value'].append(_55value[i])

    for i in range(7):
        dict['x_value'].append(60)
        dict['y_value'].append(_60value[i])


    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Probability of vehicle 1 merging first over all sessions')

    df1 = pd.DataFrame.from_dict(dict)

    sns.regplot(x="x_value", y="y_value", data=df1)
    ax.set(xlabel='Velocity vehicle 1 [km/h]', ylabel='Probability [%]')

    r, p = stats.pearsonr(df1['x_value'], df1['y_value'])

    ax.plot([], [], ' ', label='r: ' + str(round(r, 2)))
    ax.plot([], [], ' ', label='p: ' + str(round(p, 2)))
    ax.legend(loc='best')

    plt.show()
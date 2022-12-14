import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
from ast import literal_eval

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


def compute_average(folder_with_csv, condition, side):
    trails = []
    for i in range(len(folder_with_csv)):
        path_per_c = []
        for file in Path(folder_with_csv[i]).glob('*.csv'):
            path_per_c.append(file)
        trails.append(path_per_c)

    dict = {'merge_or_not': [], 'velocity': []}

    for i in range(len(trails)):
        single_folder = trails[i]
        for file in range(len(single_folder)):
            c_p = probability_calc(str(single_folder[file]), side)
            # print(c_p)
            dict['merge_or_not'].append(c_p)
            dict['velocity'] = [int(condition)] * len(dict['merge_or_not'])

    return dict


if __name__ == '__main__':
    # experiment1
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment1'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment1'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment1'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment1'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment1'

    session1_40 = compute_average([c40_60], '40', 'right')
    session1_45 = compute_average([c45_55], '45', 'right')
    session1_50 = compute_average([c50_50], '50', 'right')
    session1_55 = compute_average([c55_45], '55', 'left')
    session1_60 = compute_average([c60_40], '60', 'left')

    df1_40 = pd.DataFrame(session1_40)
    df1_45 = pd.DataFrame(session1_45)
    df1_50 = pd.DataFrame(session1_50)
    df1_55 = pd.DataFrame(session1_55)
    df1_60 = pd.DataFrame(session1_60)

    # experiment2
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment2'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment2'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment2'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment2'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment2'

    session2_40 = compute_average([c40_60], '40', 'right')
    session2_45 = compute_average([c45_55], '45', 'right')
    session2_50 = compute_average([c50_50], '50', 'right')
    session2_55 = compute_average([c55_45], '55', 'left')
    session2_60 = compute_average([c60_40], '60', 'left')

    df2_40 = pd.DataFrame(session2_40)
    df2_45 = pd.DataFrame(session2_45)
    df2_50 = pd.DataFrame(session2_50)
    df2_55 = pd.DataFrame(session2_55)
    df2_60 = pd.DataFrame(session2_60)

    #experiment3
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment3'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment3'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment3'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment3'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment3'

    session3_40 = compute_average([c40_60], '40', 'right')
    session3_45 = compute_average([c45_55], '45', 'right')
    session3_50 = compute_average([c50_50], '50', 'right')
    session3_55 = compute_average([c55_45], '55', 'left')
    session3_60 = compute_average([c60_40], '60', 'left')

    df3_40 = pd.DataFrame(session3_40)
    df3_45 = pd.DataFrame(session3_45)
    df3_50 = pd.DataFrame(session3_50)
    df3_55 = pd.DataFrame(session3_55)
    df3_60 = pd.DataFrame(session3_60)

    #experiment4
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment4'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment4'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment4'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment4'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment4'

    session4_40 = compute_average([c40_60], '40', 'right')
    session4_45 = compute_average([c45_55], '45', 'right')
    session4_50 = compute_average([c50_50], '50', 'right')
    session4_55 = compute_average([c55_45], '55', 'left')
    session4_60 = compute_average([c60_40], '60', 'left')

    df4_40 = pd.DataFrame(session4_40)
    df4_45 = pd.DataFrame(session4_45)
    df4_50 = pd.DataFrame(session4_50)
    df4_55 = pd.DataFrame(session4_55)
    df4_60 = pd.DataFrame(session4_60)

    #experiment5
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment5'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment5'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment5'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment5'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment5'

    session5_40 = compute_average([c40_60], '40', 'right')
    session5_45 = compute_average([c45_55], '45', 'right')
    session5_50 = compute_average([c50_50], '50', 'right')
    session5_55 = compute_average([c55_45], '55', 'left')
    session5_60 = compute_average([c60_40], '60', 'left')

    df5_40 = pd.DataFrame(session5_40)
    df5_45 = pd.DataFrame(session5_45)
    df5_50 = pd.DataFrame(session5_50)
    df5_55 = pd.DataFrame(session5_55)
    df5_60 = pd.DataFrame(session5_60)

    #experiment6
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment6'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment6'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment6'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment6'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment6'

    session6_40 = compute_average([c40_60], '40', 'right')
    session6_45 = compute_average([c45_55], '45', 'right')
    session6_50 = compute_average([c50_50], '50', 'right')
    session6_55 = compute_average([c55_45], '55', 'left')
    session6_60 = compute_average([c60_40], '60', 'left')

    df6_40 = pd.DataFrame(session6_40)
    df6_45 = pd.DataFrame(session6_45)
    df6_50 = pd.DataFrame(session6_50)
    df6_55 = pd.DataFrame(session6_55)
    df6_60 = pd.DataFrame(session6_60)

    #experiment7
    c40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle1\experiment7'
    c45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment7'
    c50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_50_50\experiment7'
    c55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_55_45\vehicle2\experiment7'
    c60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_pair_wise\whos_ahead_60_40\vehicle2\experiment7'

    session7_40 = compute_average([c40_60], '40', 'right')
    session7_45 = compute_average([c45_55], '45', 'right')
    session7_50 = compute_average([c50_50], '50', 'right')
    session7_55 = compute_average([c55_45], '55', 'left')
    session7_60 = compute_average([c60_40], '60', 'left')

    df7_40 = pd.DataFrame(session7_40)
    df7_45 = pd.DataFrame(session7_45)
    df7_50 = pd.DataFrame(session7_50)
    df7_55 = pd.DataFrame(session7_55)
    df7_60 = pd.DataFrame(session7_60)

    total_df = pd.concat([df1_40, df2_40, df3_40, df4_40, df5_40, df6_40, df7_40,
                          df1_45, df2_45, df3_45, df4_45, df5_45, df6_45, df7_45,
                          df1_50, df2_50, df3_50, df4_50, df5_50, df6_50, df7_50,
                          df1_55, df2_55, df3_55, df4_55, df5_55, df6_55, df7_55,
                          df1_60, df2_60, df3_60, df4_60, df5_60, df6_60, df7_60], ignore_index=True)

    lr = pg.logistic_regression(total_df['velocity'], total_df['merge_or_not']).round(3)
    print(lr)

    p_velocity40 = 1 / (1 + np.exp(-4.779))
    p_velocity45 = 1 / (1 + np.exp(-(4.779-0.100)))
    p_velocity50 = 1 / (1 + np.exp(-(4.679-0.100)))
    p_velocity55 = 1 / (1 + np.exp(-(4.579-0.100)))
    p_velocity60 = 1 / (1 + np.exp(-(4.479-0.100)))

    print(p_velocity40, p_velocity45, p_velocity50, p_velocity55, p_velocity60)
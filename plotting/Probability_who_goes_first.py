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

    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=110,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=215,
                                               track_start_point_distance=430,
                                               track_section_length_before=304.056,
                                               track_section_length_after=200)  # goes until 400

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

    if left_or_right == 'left':
        if index_of_mergepoint_vehicle2 < index_of_mergepoint_vehicle1:
            return 1
        else:
            return 0

    if left_or_right == 'right':
        if index_of_mergepoint_vehicle1 < index_of_mergepoint_vehicle2:
            return 1
        else:
            return 0

    if left_or_right == 'equal_v1':
        if index_of_mergepoint_vehicle1 < index_of_mergepoint_vehicle2:
            return 1
        else:
            return 0

    if left_or_right == 'equal_v2':
        if index_of_mergepoint_vehicle2 < index_of_mergepoint_vehicle1:
            return 1
        else:
            return 0

if __name__ == '__main__':

    # 60_40 left ahead
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\left'
    left_or_right = 'left'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_60_40 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_60_40.append(p1)
    probability_60_40.append(p2)
    probability_60_40.append(p3)
    probability_60_40.append(p4)
    probability_60_40.append(p5)
    probability_60_40.append(p6)
    probability_60_40.append(p7)

    nested_probability = list(average_nested(probability_60_40))
    df1 = pd.DataFrame({'probability_60_40': nested_probability})

    # 55-45 left ahead
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\left'
    left_or_right = 'left'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_55_45 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_55_45.append(p1)
    probability_55_45.append(p2)
    probability_55_45.append(p3)
    probability_55_45.append(p4)
    probability_55_45.append(p5)
    probability_55_45.append(p6)
    probability_55_45.append(p7)

    nested_probability = list(average_nested(probability_55_45))
    df2 = pd.DataFrame({'probability_55_45': nested_probability})

    # 50_50
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    left_or_right = 'equal_v2'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_50_50 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_50_50.append(p1)
    probability_50_50.append(p2)
    probability_50_50.append(p3)
    probability_50_50.append(p4)
    probability_50_50.append(p5)
    probability_50_50.append(p6)
    probability_50_50.append(p7)

    nested_probability = list(average_nested(probability_50_50))
    df3 = pd.DataFrame({'probability_50_50': nested_probability})

    left_probability = pd.concat([df1, df2, df3], axis=1)

    # 40_60 right ahead
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\right'
    left_or_right = 'right'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_40_60 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_40_60.append(p1)
    probability_40_60.append(p2)
    probability_40_60.append(p3)
    probability_40_60.append(p4)
    probability_40_60.append(p5)
    probability_40_60.append(p6)
    probability_40_60.append(p7)

    nested_probability = list(average_nested(probability_40_60))
    df4 = pd.DataFrame({'probability_40_60': nested_probability})

    # 45_ 55 right ahead
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\right'
    left_or_right = 'right'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_45_55 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_45_55.append(p1)
    probability_45_55.append(p2)
    probability_45_55.append(p3)
    probability_45_55.append(p4)
    probability_45_55.append(p5)
    probability_45_55.append(p6)
    probability_45_55.append(p7)

    nested_probability = list(average_nested(probability_45_55))
    df5 = pd.DataFrame({'probability_45_55': nested_probability})

    # 50_50
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    left_or_right = 'equal_v1'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    probability_50_50_v1 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0, 8):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p1.append(p_vehicle)
    for i in range(8, 16):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p2.append(p_vehicle)
    for i in range(16, 24):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p3.append(p_vehicle)
    for i in range(24, 32):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p4.append(p_vehicle)
    for i in range(32, 40):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p5.append(p_vehicle)
    for i in range(40, 48):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p6.append(p_vehicle)
    for i in range(48, 56):
        p_vehicle = probability_calc(trails[i], left_or_right)
        p7.append(p_vehicle)
    probability_50_50_v1.append(p1)
    probability_50_50_v1.append(p2)
    probability_50_50_v1.append(p3)
    probability_50_50_v1.append(p4)
    probability_50_50_v1.append(p5)
    probability_50_50_v1.append(p6)
    probability_50_50_v1.append(p7)

    nested_probability = list(average_nested(probability_50_50_v1))
    df6 = pd.DataFrame({'probability_50_50': nested_probability})

    right_probability = pd.concat([df4, df5, df6], axis=1)

    print(left_probability)
    print(right_probability)

    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'red'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Comparison probability merging first on headway advantage')
    sns.boxplot(ax=axes[0], data=right_probability, **PROPS)
    sns.boxplot(ax=axes[1], data=left_probability, **PROPS)
    axes[0].set_title('Vehicle 1 positive headway')
    axes[1].set_title('Vehicle 2 positive headway')

    plt.show()











import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

import os
import pickle
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


# if __name__ == '__main__':
def plot_trail(path_to_data_csv):
    # data = pd.read_csv(
    #     'C:\\Users\localadmin\Desktop\Joan_testdata_CRT\joan_data_20220901_14h00m40s.csv',
    #     sep=';')
    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
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

    data_dict = {'y1_straight': [],
                 'y2_straight': [],
                 }

    if xy_coordinates_vehicle1[0][0] > 0:
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

    elif xy_coordinates_vehicle2[0][0] > 0:
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])


    index_of_mergepoint_vehicle1 = min(range(len(data_dict['y1_straight'])),
                                       key=lambda i: abs(data_dict['y1_straight'][i] - track.merge_point[1]))
    index_of_mergepoint_vehicle2 = min(range(len(data_dict['y2_straight'])),
                                       key=lambda i: abs(data_dict['y2_straight'][i] - track.merge_point[1]))


    #
    #
    # # final plotting
    # # ax1.plot([], [], ' ', label='crt: ' + str(round(crt, 2)))
    # ax1.legend(loc='upper right', prop={'size': 8})
    # leg = ax1.get_legend()
    # for i in range(2, 5):
    #     leg.legendHandles[i].set_color('black')


    plt.show()

    # a_file = open("global_data_dict.pkl", "wb")
    # pickle.dump(data_dict, a_file)
    # a_file.close()


if __name__ == '__main__':
    # 55-45
    files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    index = 1
    plot_trail(trails[index])
    # #
    # for i in range(len(trails)):
    #     plot_trail(trails[5+i])


    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i])
    #     plt.savefig(r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_55_45\figures\condition_55_45_trail_{}'.format(str(figure_amount)))
    #     figure_amount += 1
    #
    # ## 50-50
    # files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50'
    # trails = []
    # for file in Path(files_directory).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails.append(file)
    # trails = natsorted(trails, key=str)
    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i])
    #     plt.savefig(r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50\figures\condition_50_50_trail_{}'.format(str(figure_amount)))
    #     figure_amount += 1
    #
    # ##60-40
    # files_directory = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40'
    # trails = []
    # for file in Path(files_directory).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails.append(file)
    # trails = natsorted(trails, key=str)
    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i])
    #     plt.savefig(r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40\figures\condition_60_40_trail_{}'.format(str(figure_amount)))
    #     figure_amount += 1


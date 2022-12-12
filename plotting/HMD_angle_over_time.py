import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
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

def get_timestamps(intcolumnname, data_csv):
    time = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data_csv.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time

def hmd_over_time(path_to_data_csv, condition):

    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=135,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=150)

    track = SymmetricMergingTrack(simulation_constants)

    data_dict = {'time': [],
                 'velocity_vehicle1': [],
                 'velocity_vehicle2': [],
                 'y1_straight': [],
                 'y2_straight': [],
                 'distance_traveled_vehicle1': [],
                 'distance_traveled_vehicle2': [],
                 }

    xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
    xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, data)

    xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
    xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

    xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
    xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)

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

    # ------------------------------------------------------------------------#
    # time
    time_in_datetime = get_timestamps(0, data)
    time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
    time_in_seconds_trail = np.array(time_in_seconds_trail)
    data_dict['time'] = time_in_seconds_trail

    # ------------------------------------------------------------------------#
    # average travelled
    for i in range(len(xy_coordinates_vehicle1)):
        traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
        traveled_distance2 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle2[i])
        data_dict['distance_traveled_vehicle1'].append(traveled_distance1)
        data_dict['distance_traveled_vehicle2'].append(traveled_distance2)

    data_dict['distance_traveled_vehicle1'] = gaussian_filter1d(data_dict['distance_traveled_vehicle1'], sigma=15)
    data_dict['distance_traveled_vehicle2'] = gaussian_filter1d(data_dict['distance_traveled_vehicle2'], sigma=15)

    index_of_tunnel_vehicle1 = min(range(len(data_dict['distance_traveled_vehicle1'])),
                                   key=lambda i: abs(data_dict['distance_traveled_vehicle1'][i] - track.tunnel_length))
    index_of_tunnel_vehicle2 = min(range(len(data_dict['distance_traveled_vehicle2'])),
                                   key=lambda i: abs(data_dict['distance_traveled_vehicle2'][i] - track.tunnel_length))

    index_of_mergepoint_vehicle1 = min(range(len(data_dict['y1_straight'])),
                                       key=lambda i: abs(data_dict['y1_straight'][i] - track.merge_point[1]))
    index_of_mergepoint_vehicle2 = min(range(len(data_dict['y2_straight'])),
                                       key=lambda i: abs(
                                           data_dict['y2_straight'][i] - track.merge_point[1]))

    who_is_first_tunnel = min(index_of_tunnel_vehicle1, index_of_tunnel_vehicle2)
    who_is_first_merge = min(index_of_mergepoint_vehicle1, index_of_mergepoint_vehicle2)


    v1 = list(data['HMD_rotation_vehicle1'][who_is_first_tunnel:who_is_first_merge])
    v2 = list(data['HMD_rotation_vehicle2'][who_is_first_tunnel:who_is_first_merge])
    x = data_dict['time'][who_is_first_tunnel:who_is_first_merge]

    plt.plot(x, v1, label = 'vehicle 1')
    plt.plot(x, v2, label = 'vehicle 2')
    plt.xlabel('Time [s]')
    plt.ylabel('HMD rotation [-]')

    if condition == '50-50':
        plt.title('Head mount rotation over time during the interactive approach\n Condition 1 (50-50 km/h)')
    if condition == '55-45':
        plt.title('Head mount rotation over time during the interactive approach\n Condition 2 (55-45 km/h)')
    if condition == '60-40':
        plt.title('Head mount rotation over time during the interactive approach\n Condition 3 (60-40 km/h)')

    plt.legend()
    # plt.show()


if __name__ == '__main__':
    files_directory = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    files_directory1 = r'D:\Thesis_data_all_experiments\Conditions\condition_55_45'
    files_directory2 = r'D:\Thesis_data_all_experiments\Conditions\condition_60_40'

    # trails = []
    # for file in Path(files_directory).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails.append(file)
    # trails = natsorted(trails, key=str)
    #
    # figure_amount = 0
    # condition1 = '50-50'
    # for i in range(len(trails)):
    #     hmd_over_time(trails[i], condition1)
    #     fig = plt.savefig(
    #         r'D:\Thesis_data_all_experiments\Figures_hmd_rots\Condition 1 - 50_50\condition1_trail_{}'.format(
    #             str(figure_amount)))
    #     plt.close(fig)
    #     figure_amount += 1
    #
    # trails1 = []
    # for file in Path(files_directory1).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails1.append(file)
    # trails1 = natsorted(trails1, key=str)
    #
    # figure_amount = 0
    # condition2 = '55-45'
    # for i in range(len(trails1)):
    #     hmd_over_time(trails1[i], condition2)
    #     fig = plt.savefig(
    #         r'D:\Thesis_data_all_experiments\Figures_hmd_rots\Condition 2 - 55_45\condition2_trail_{}'.format(
    #             str(figure_amount)))
    #     plt.close(fig)
    #     figure_amount += 1

    trails2 = []
    for file in Path(files_directory2).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails2.append(file)
    trails2 = natsorted(trails2, key=str)

    figure_amount = 0
    condition3 = '60-40'
    for i in range(len(trails2)):
        hmd_over_time(trails2[i], condition3)
        fig = plt.savefig(
            r'D:\Thesis_data_all_experiments\Figures_hmd_rots\Condition 3 - 60_40\condition3_trail_{}'.format(
                str(figure_amount)))
        plt.close(fig)
        figure_amount += 1


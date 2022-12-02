import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
import os
import pickle
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot

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


def vehicle_velocity(intcolumnname, data_csv):
    velocity = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        velocity_vehicle = eval(data_csv.iloc[i, intcolumnname])
        x_loc = velocity_vehicle[0]
        velocity.append(x_loc)
        # velocity = list(dict.fromkeys(velocity))
    return velocity


def get_timestamps(intcolumnname, data_csv):
    time = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data_csv.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time


def check_if_on_collision_course_for_point(travelled_distance_collision_point, data_dict, simulation_constants):
    track = SymmetricMergingTrack(simulation_constants)
    point_predictions = {'vehicle1': [], 'vehicle2': []}

    point_predictions['vehicle1'] = np.array(data_dict['distance_traveled_vehicle1']) + np.array(
        data_dict['velocity_vehicle1']) * (travelled_distance_collision_point - np.array(
        data_dict['distance_traveled_vehicle2'])) / np.array(
        data_dict['velocity_vehicle2'])
    point_predictions['vehicle2'] = np.array(data_dict['distance_traveled_vehicle2']) + np.array(
        data_dict['velocity_vehicle2']) * (travelled_distance_collision_point - np.array(
        data_dict['distance_traveled_vehicle1'])) / np.array(
        data_dict['velocity_vehicle1'])

    lb, ub = track.get_collision_bounds(travelled_distance_collision_point, simulation_constants.vehicle_width,
                                        simulation_constants.vehicle_length)

    on_collision_course = ((lb < point_predictions['vehicle1']) & (point_predictions['vehicle1'] < ub)) | \
                          ((lb < point_predictions['vehicle2']) & (point_predictions['vehicle2'] < ub))

    return on_collision_course[0]


def calculate_conflict_resolved_time(data_dict, simulation_constants, condition):
    time = data_dict['time']
    track = SymmetricMergingTrack(simulation_constants)

    merge_point_collision_course = check_if_on_collision_course_for_point(track.section_length_before, data_dict,
                                                                          simulation_constants)

    threshold_collision_course = check_if_on_collision_course_for_point(track.upper_bound_threshold + 1e-3, data_dict,
                                                                        simulation_constants)
    # 1e-3 is used for straight approach (always inside)
    end_point_collision_course = check_if_on_collision_course_for_point(
        track.section_length_before + track.section_length_after,
        data_dict, simulation_constants)

    on_collision_course = merge_point_collision_course \
                          | threshold_collision_course \
                          | end_point_collision_course

    # on_collision_course = merge_point_collision_course \
    #                       | end_point_collision_course

    if condition == '50-50':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 304)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 304))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '55-45':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 304)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 304))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '60-40':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 290)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 290))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    # if condition == '60-40':
    #     approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle1']) < 290)) | \
    #                     ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle2']) < 305))
    #     indices_of_conflict_resolved = (on_collision_course & approach_mask)

    # if condition == '50-50':
    #     approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle1']) < 305)) | \
    #                     ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle2']) < 305))
    #     indices_of_conflict_resolved = (on_collision_course & approach_mask)
    #
    # elif condition == '55-45':
    #     approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle1']) < 305)) | \
    #                     ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
    #                      (np.array(data_dict['distance_traveled_vehicle2']) < 305))
    #     indices_of_conflict_resolved = (on_collision_course & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


# if __name__ == '__main__':
def plot_trail(path_to_data_csv, headway_bounds, condition):
    # data = pd.read_csv(
    #     'C:\\Users\localadmin\Desktop\Joan_testdata_CRT\joan_data_20220901_14h00m40s.csv',
    #     sep=';')
    data = pd.read_csv(path_to_data_csv, sep=',')
    # print(data.iloc[:,1])
    # print(data.iloc[:,1][data.iloc[:,1]].index.values)
    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    # simulation_constants = SimulationConstants(vehicle_width=1.5,
    #                                            vehicle_length=4.7,
    #                                            tunnel_length=118,  # original = 118 -> check in unreal
    #                                            track_width=8,
    #                                            track_height=215,
    #                                            track_start_point_distance=430,
    #                                            track_section_length_before=304,
    #                                            track_section_length_after=145)

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=135,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)

    xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
    xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, data)

    xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
    xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

    xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
    xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)

    data_dict = {'time': [],
                 'interactive_time': [],
                 'x1_straight': [],
                 'y1_straight': [],
                 'x2_straight': [],
                 'y2_straight': [],
                 'velocity_vehicle1': [],
                 'velocity_vehicle2': [],
                 'distance_traveled_vehicle1': [],
                 'distance_traveled_vehicle2': [],
                 'average_travelled_distance_trace': [],
                 'headway': []}

    if xy_coordinates_vehicle1[0][0] > 0:
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['x1_straight'].append(straight_line_vehicle1[0][0] + 15)
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['x2_straight'].append(straight_line_vehicle2[0][0] - 15)
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

    elif xy_coordinates_vehicle2[0][0] > 0:
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['x1_straight'].append(straight_line_vehicle1[0][0] - 15)
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['x2_straight'].append(straight_line_vehicle2[0][0] + 15)
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    # fig.suptitle('-')
    # plt.title('positions over time')
    ax1.set_xlabel('y position [m]')
    ax1.set_ylabel('x position [m]')
    ax1.set_yticks([175, -20, 20, -175])
    ax1.set_yticklabels([175, 0, 0, -175])
    ax1.set_xlim(60, 450)
    ax1.set_ylim(-200, 200)

    right_positions_per_second = list(zip(data_dict['x1_straight'], data_dict['y1_straight']))[0::300]
    left_positions_per_second = list(zip(data_dict['x2_straight'], data_dict['y2_straight']))[0::300]

    for right_point, left_point in zip(left_positions_per_second, right_positions_per_second):
        ax1.plot([left_point[1], right_point[1]], [left_point[0], right_point[0]], c='lightgrey', linestyle='dashed')
        ax1.scatter([left_point[1], right_point[1]], [left_point[0], right_point[0]], c='lightgrey')

    ax1.plot(data_dict['y1_straight'], data_dict['x1_straight'], label='right vehicle')
    ax1.plot(data_dict['y2_straight'], data_dict['x2_straight'], label='left vehicle')

    # ------------------------------------------------------------------------#
    # velocity_time plot
    velocity_vehicle1 = vehicle_velocity(3, data)
    velocity_vehicle2 = vehicle_velocity(6, data)

    for i in range(len(velocity_vehicle1)):
        data_dict['velocity_vehicle1'].append(velocity_vehicle1)

        data_dict['velocity_vehicle2'].append(velocity_vehicle2)

    time_in_datetime = get_timestamps(0, data)
    time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
    time_in_seconds_trail = np.array(time_in_seconds_trail)
    data_dict['time'] = time_in_seconds_trail

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_xlim(0, 30)

    ax2.plot(data_dict['time'], velocity_vehicle1)
    ax2.plot(data_dict['time'], velocity_vehicle2)

    fig.tight_layout(pad=1.0)

    # ------------------------------------------------------------------------#
    # average travelled against headway
    traveled_distance1 = []
    traveled_distance2 = []
    for i in range(len(xy_coordinates_vehicle1)):
        _traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
        _traveled_distance2 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle2[i])
        traveled_distance1.append(_traveled_distance1)
        traveled_distance2.append(_traveled_distance2)

    data_dict['distance_traveled_vehicle1'] = gaussian_filter1d(traveled_distance1, sigma=15)
    data_dict['distance_traveled_vehicle2'] = gaussian_filter1d(traveled_distance2, sigma=15)

    average_travelled_distance_trace = list((np.array(data_dict['distance_traveled_vehicle1']) + np.array(
        data_dict['distance_traveled_vehicle2'])) / 2.)

    for i in range(len(average_travelled_distance_trace)):
        data_dict['average_travelled_distance_trace'].append(average_travelled_distance_trace[i])

    headway = np.array(data_dict['distance_traveled_vehicle1']) - np.array(data_dict['distance_traveled_vehicle2'])

    for i in range(len(headway)):
        data_dict['headway'].append(headway[i])

    ax3.plot(average_travelled_distance_trace, headway, c='lightgray')
    ax3.set_xlabel('Average travelled distance [m]')
    ax3.set_ylabel('Headway [m]')
    ax3.set_xlim(50, 450)
    fig.tight_layout(pad=1.0)

    # plot merge point
    index_of_mergepoint_vehicle1 = min(range(len(data_dict['y1_straight'])),
                                       key=lambda i: abs(data_dict['y1_straight'][i] - 230))
    index_of_mergepoint_vehicle2 = min(range(len(data_dict['y2_straight'])),
                                       key=lambda i: abs(
                                           data_dict['y2_straight'][i] - 230))  # instead of track.mergepoint

    if xy_coordinates_vehicle1[0][0] > 0:
        ax1.scatter(track.merge_point[1], track.merge_point[0] + 15, c='purple', marker='s', s=30, zorder=2,
                    label='merge point')
        ax1.scatter(track.merge_point[1], track.merge_point[0] - 15, c='orange', marker='s', s=30, zorder=2)
    elif xy_coordinates_vehicle2[0][0] > 0:
        ax1.scatter(track.merge_point[1], track.merge_point[0] - 15, c='purple', marker='s', s=30, zorder=2,
                    label='merge point')
        ax1.scatter(track.merge_point[1], track.merge_point[0] + 15, c='orange', marker='s', s=30, zorder=2)

    ax2.scatter(data_dict['time'][index_of_mergepoint_vehicle1], velocity_vehicle1[index_of_mergepoint_vehicle1],
                c='purple', marker='s', s=30, zorder=2)
    ax2.scatter(data_dict['time'][index_of_mergepoint_vehicle2], velocity_vehicle2[index_of_mergepoint_vehicle2],
                c='orange', marker='s', s=30, zorder=2)

    ax3.scatter(average_travelled_distance_trace[index_of_mergepoint_vehicle1], headway[index_of_mergepoint_vehicle1],
                c='purple', marker='s', s=30, zorder=2)
    ax3.scatter(average_travelled_distance_trace[index_of_mergepoint_vehicle2], headway[index_of_mergepoint_vehicle2],
                c='orange', marker='s', s=30, zorder=2)

    # plot tunnel exit
    index_of_tunnel_vehicle1 = min(range(len(data_dict['distance_traveled_vehicle1'])),
                                   key=lambda i: abs(data_dict['distance_traveled_vehicle1'][i] - track.tunnel_length))
    index_of_tunnel_vehicle2 = min(range(len(data_dict['distance_traveled_vehicle2'])),
                                   key=lambda i: abs(data_dict['distance_traveled_vehicle2'][i] - track.tunnel_length))

    ax1.scatter(data_dict['y1_straight'][index_of_tunnel_vehicle1], data_dict['x1_straight'][index_of_tunnel_vehicle1],
                c='purple', marker='>', s=50, zorder=2, label='tunnel exit')
    ax1.scatter(data_dict['y2_straight'][index_of_tunnel_vehicle2], data_dict['x2_straight'][index_of_tunnel_vehicle2],
                c='orange', marker='>', s=50, zorder=2)

    ax2.scatter(data_dict['time'][index_of_tunnel_vehicle1], velocity_vehicle1[index_of_tunnel_vehicle1],
                c='purple', marker='>', s=30, zorder=2)
    ax2.scatter(data_dict['time'][index_of_tunnel_vehicle2], velocity_vehicle2[index_of_tunnel_vehicle2],
                c='orange', marker='>', s=30, zorder=2)

    ax3.scatter(data_dict['average_travelled_distance_trace'][index_of_tunnel_vehicle1],
                data_dict['headway'][index_of_tunnel_vehicle1], c='purple', marker='>', s=30, zorder=2)

    ax3.scatter(data_dict['average_travelled_distance_trace'][index_of_tunnel_vehicle2],
                data_dict['headway'][index_of_tunnel_vehicle2], c='orange', marker='>', s=30, zorder=2)

    who_is_first_tunnel = min(index_of_tunnel_vehicle1, index_of_tunnel_vehicle2)
    who_is_first_tunnel_time = time_in_seconds_trail[who_is_first_tunnel]

    ##compute/plot crt
    crt_object = calculate_conflict_resolved_time(data_dict, simulation_constants, condition)

    if not crt_object[1].size:
        crt = 0
    else:
        crt = crt_object[1][-1]

    if not crt_object[1].size:
        index_crt = 0
    else:
        index_crt = min(range(len(data_dict['time'])),
                        key=lambda i: abs(data_dict['time'][i] - crt))

    ax1.scatter(data_dict['y1_straight'][index_crt], data_dict['x1_straight'][index_crt], c='black', marker='x', s=50,
                zorder=2, label='crt: ' + str(round(crt - who_is_first_tunnel_time, 2)) + ' sec')
    ax1.scatter(data_dict['y2_straight'][index_crt], data_dict['x2_straight'][index_crt], c='black', marker='x',
                s=50, zorder=2)

    ax2.scatter(data_dict['time'][index_crt], velocity_vehicle1[index_crt], c='black', marker='x', s=50, zorder=2)
    ax2.scatter(data_dict['time'][index_crt], velocity_vehicle2[index_crt], c='black', marker='x', s=50, zorder=2)

    ax3.scatter(average_travelled_distance_trace[index_crt], headway[index_crt], c='black', marker='x', s=50, zorder=2)

    # collision area
    ax3.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object),
             np.array(headway_bounds['negative_headway_bound'], dtype=object),
             linestyle='dashed', c='grey')
    ax3.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object),
             np.array(headway_bounds['positive_headway_bound'], dtype=object),
             linestyle='dashed', c='grey')

    ax3.fill_between(headway_bounds['average_travelled_distance'], headway_bounds['negative_headway_bound'],
                     headway_bounds['positive_headway_bound'],
                     color='lightgrey')

    ax3.text(350, 0., 'Collision area', verticalalignment='center', clip_on=True)

    # final plotting
    ax1.text(-0.08, 0.42, 'A)', transform=ax1.transAxes,
             fontsize=12, va='bottom', ha='right')
    ax2.text(-0.08, 0.42, 'B)', transform=ax2.transAxes,
             fontsize=12, va='bottom', ha='right')
    ax3.text(-0.08, 0.42, 'C)', transform=ax3.transAxes,
             fontsize=12, va='bottom', ha='right')

    ax1.legend(loc='upper right', prop={'size': 8})
    leg = ax1.get_legend()
    for i in range(2, 5):
        leg.legendHandles[i].set_color('black')

    fig.set_size_inches(10, 6)

    # plt.show()

    #
    # a_file = open("global_data_dict.pkl", "wb")
    # pickle.dump(data_dict, a_file)
    # a_file.close()


if __name__ == '__main__':
    with open(os.path.join('..', 'data_folder', 'headway_bounds.pkl'), 'rb') as f:
        headway_bounds = pickle.load(f)

    # 55-45
    files_directory = r'D:\Thesis_data_all_experiments\Conditions\condition_55_45'
    condition = '55-45'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    index = 15
    plot_trail(trails[index], headway_bounds, condition)

    # for i in range(len(trails)):
    #     plot_trail(trails[i], headway_bounds, condition)
    #
    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i], headway_bounds, condition)
    #     fig = plt.savefig(
    #         r'D:\Thesis_data_all_experiments\Conditions\condition_55_45\figures\condition_55_45_trail_{}'.format(
    #             str(figure_amount)))
    #     plt.close(fig)
    #     figure_amount += 1
    #
    # ## 50-50
    # files_directory = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    # condition = '50-50'
    # trails = []
    # for file in Path(files_directory).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails.append(file)
    # trails = natsorted(trails, key=str)
    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i], headway_bounds, condition)
    #     fig = plt.savefig(
    #         r'D:\Thesis_data_all_experiments\Conditions\condition_50_50\figures\condition_50_50_trail_{}'.format(
    #             str(figure_amount)))
    #     plt.close(fig)
    #     figure_amount += 1
    #
    # ##60-40
    # files_directory = r'D:\Thesis_data_all_experiments\Conditions\condition_60_40'
    # condition = '60-40'
    # trails = []
    # for file in Path(files_directory).glob('*.csv'):
    #     # trail_condition = plot_trail(file)
    #     trails.append(file)
    # trails = natsorted(trails, key=str)
    # figure_amount = 0
    # for i in range(len(trails)):
    #     plot_trail(trails[i], headway_bounds, condition)
    #     fig = plt.savefig(
    #         r'D:\Thesis_data_all_experiments\Conditions\condition_60_40\figures\condition_60_40_trail_{}'.format(
    #             str(figure_amount)))
    #     plt.close(fig)
    #     figure_amount += 1

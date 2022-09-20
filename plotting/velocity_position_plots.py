import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
# import seaborn as sns
import pickle
import datetime
import tqdm
from pathlib import Path
import statistics

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

    on_collision_course = ((lb < point_predictions['vehicle2']) & (point_predictions['vehicle2'] < ub)) | \
                          ((lb < point_predictions['vehicle1']) & (point_predictions['vehicle1'] < ub))

    return on_collision_course[0]


def calculate_conflict_resolved_time(data_dict, simulation_constants):
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

    if 11 < sum(data_dict['velocity_vehicle1'][0][0:10]) / len(data_dict['velocity_vehicle1'][0][0:10]) < 11.4:
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < track.section_length_before)) |\
                         ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < track.section_length_before))
        indices_of_conflict_resolved = (on_collision_course == False & approach_mask)

    elif sum(data_dict['velocity_vehicle1'][0][0:10]) / len(data_dict['velocity_vehicle1'][0][0:10]) > 11.5:
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < track.section_length_before))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    elif sum(data_dict['velocity_vehicle1'][0][0:10]) / len(data_dict['velocity_vehicle1'][0][0:10]) < 11:
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < track.section_length_before))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)


    # # approach_mask = (approach_mask == False)
    #
    # ###PROBLEM IS HERE
    # # x = [i for i, x in enumerate(approach_mask) if x]
    # # # ->https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
    # print(list(on_collision_course))
    # # print(list(approach_mask))
    # print(list(approach_mask == False))
    # # print(x)
    # indices_of_conflict_resolved = (on_collision_course == False & approach_mask)
    # print(list(indices_of_conflict_resolved))

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


# if __name__ == '__main__':
def plot_trail(path_to_data_csv, left_or_right_ahead):
    # data = pd.read_csv(
    #     'C:\\Users\localadmin\Desktop\Joan_testdata_CRT\joan_data_20220901_14h00m40s.csv',
    #     sep=';')
    data = pd.read_csv(path_to_data_csv, sep=';')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=118,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=200)  # goes until 400

    track = SymmetricMergingTrack(simulation_constants)

    xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
    xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, data)

    xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
    xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

    xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
    xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)

    data_dict = {'time': [],
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

    if left_or_right_ahead == 'right':
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['x1_straight'].append(straight_line_vehicle1[0][0] + 5)
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['x2_straight'].append(straight_line_vehicle2[0][0] - 5)
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

    elif left_or_right_ahead == 'left':
        for i in range(len(xy_coordinates_vehicle1)):
            straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
            data_dict['x1_straight'].append(straight_line_vehicle1[0][0] - 5)
            data_dict['y1_straight'].append(straight_line_vehicle1[0][1])

        for i in range(len(xy_coordinates_vehicle2)):
            straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
            data_dict['x2_straight'].append(straight_line_vehicle2[0][0] + 5)
            data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    # fig.suptitle('-')
    # plt.title('positions over time')
    ax1.set_xlabel('y position [m]')
    ax1.set_ylabel('x position [m]')
    ax1.set_yticks([175, -5, 5, -175])
    ax1.set_yticklabels([175, 0, 0, -175])

    # ax1.tick_params(axis='y', labelsize=10)
    # ax1.tick_params(axis='x', labelsize=10)

    ax1.scatter(data_dict['y1_straight'][0::300], data_dict['x1_straight'][0::300], s=10)
    ax1.scatter(data_dict['y2_straight'][0::300], data_dict['x2_straight'][0::300], s=10)
    ax1.set_xlim(50, 450)

    ax1.plot(data_dict['y1_straight'], data_dict['x1_straight'], label='right vehicle')
    ax1.plot(data_dict['y2_straight'], data_dict['x2_straight'], label='left vehicle')
    # ax1.plot(data_dict['x1_straight'], data_dict['y1_straight'])
    # ax1.plot(data_dict['x2_straight'], data_dict['y2_straight'])

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

    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.title('velocity at times')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_xlim(0, 30)
    ax2.plot(data_dict['time'], velocity_vehicle1)
    ax2.plot(data_dict['time'], velocity_vehicle2)

    fig.tight_layout(pad=1.0)

    # ------------------------------------------------------------------------#
    # average travelled against headway

    for i in range(len(xy_coordinates_vehicle1)):
        traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
        traveled_distance2 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle2[i])
        data_dict['distance_traveled_vehicle1'].append(traveled_distance1)
        data_dict['distance_traveled_vehicle2'].append(traveled_distance2)

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

    ##compute/plot crt
    crt_object = calculate_conflict_resolved_time(data_dict, simulation_constants)
    crt = crt_object[1]

    index_crt = np.where(crt_object[0] == True)[0][0]

    ax1.scatter(data_dict['y1_straight'][index_crt], data_dict['x1_straight'][index_crt], c='purple', marker='x', s=50,
                zorder=2, label='crt')
    ax1.scatter(data_dict['y2_straight'][index_crt], data_dict['x2_straight'][index_crt], c='orange', marker='x',
                s=50, zorder=2)


    ax2.scatter(data_dict['time'][index_crt], velocity_vehicle1[index_crt], c='purple', marker='x', s=50, zorder=2)
    ax2.scatter(data_dict['time'][index_crt], velocity_vehicle2[index_crt], c='orange', marker='x', s=50, zorder=2)

    ax3.scatter(average_travelled_distance_trace[index_crt], headway[index_crt], c='black', marker='x', s=50, zorder=2)

    # plot merge point
    index_of_mergepoint_vehicle1 = min(range(len(data_dict['y1_straight'])),
                                       key=lambda i: abs(data_dict['y1_straight'][i] - track.merge_point[1]))
    index_of_mergepoint_vehicle2 = min(range(len(data_dict['y2_straight'])),
                                       key=lambda i: abs(data_dict['y2_straight'][i] - track.merge_point[1]))

    if left_or_right_ahead == 'right':
        ax1.scatter(track.merge_point[1], track.merge_point[0] + 5, c='purple', marker='s', s=30, zorder=2,
                    label='merge point')
        ax1.scatter(track.merge_point[1], track.merge_point[0] - 5, c='orange', marker='s', s=30, zorder=2)
    elif left_or_right_ahead == 'left':
        ax1.scatter(track.merge_point[1], track.merge_point[0] - 5, c='purple', marker='s', s=30, zorder=2,
                    label='merge point')
        ax1.scatter(track.merge_point[1], track.merge_point[0] + 5, c='orange', marker='s', s=30, zorder=2)

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

    # final plotting
    ax1.legend(loc='upper right')
    leg = ax1.get_legend()
    for i in range(2, 5):
        leg.legendHandles[i].set_color('black')

    # # Question for olger:
    #
    # data_dict_bounds = {'positive_headway_bound': [],
    #                     'negative_headway_bound': [],
    #                     'average_travelled_distance': []}
    #
    # for average_y_position_in_mm in tqdm.trange(int((track.section_length_before) * 1000)):
    #     average_y_position = average_y_position_in_mm / 1000.
    #
    #     lb, ub = track.get_headway_bounds(average_y_position,
    #                                       vehicle_length=simulation_constants.vehicle_length,
    #                                       vehicle_width=simulation_constants.vehicle_width)
    #
    #     data_dict_bounds['positive_headway_bound'].append(ub)
    #     data_dict_bounds['negative_headway_bound'].append(lb)
    #     data_dict_bounds['average_travelled_distance'].append(average_y_position)
    #
    # # for key in data_dict_bounds.keys():
    # #     data_dict_bounds[key] = np.array(data_dict_bounds[key], dtype=float)
    #
    #
    # ax3.plot(data_dict_bounds['average_travelled_distance'], data_dict_bounds['positive_headway_bound'], c='gray')
    # ax3.plot(data_dict_bounds['average_travelled_distance'], data_dict_bounds['negative_headway_bound'], c='gray')
    # # ax3.plot(np.array(data_dict_bounds['average_travelled_distance'], dtype=object),
    # #          np.array(data_dict_bounds['positive_headway_bound'], dtype=object),
    # #          linestyle='dashed', c='black')
    #
    #
    # ax3.fill_between(range(200,400), data_dict_bounds['negative_headway_bound'],
    #                       data_dict_bounds['positive_headway_bound'],
    #                       color='lightgrey')
    #
    # for key in data_dict.keys():
    #     data_dict[key] = np.array(data_dict[key], dtype=float)
    #
    #
    # ax3.text(290., 0, 'Collision area', verticalalignment='center', clip_on=True)

    plt.show()

    # a_file = open("global_data_dict.pkl", "wb")
    # pickle.dump(data_dict, a_file)
    # a_file.close()


if __name__ == '__main__':
    # sort condition1
    files_directory = r'C:\Users\localadmin\Desktop\ExperimentOlgerArkady'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)

    print(trails)
    # path_to_csv = 'C:\\Users\loran\Desktop\ExperimentOlgerArkady\joan_data_20220915_14h15m00s.csv'
    trail_condition2 = plot_trail(trails[2], left_or_right_ahead = 'right')  # 46-34
    trail_condition6 = plot_trail(trails[6], left_or_right_ahead = 'right')  # "43-37",
    trail_condition8 = plot_trail(trails[8], left_or_right_ahead = 'right')  # "equal40-40",
    trail_condition10 = plot_trail(trails[10], left_or_right_ahead = 'right')  # "34-46",
    trail_condition12 = plot_trail(trails[12], left_or_right_ahead = 'right')  # "37-43",

    trail_condition3 = plot_trail(trails[3], left_or_right_ahead = 'left') #equal40-40-flipped-side
    # trail_condition = plot_trail(trails[5], left_or_right_ahead = 'left') #"46-34-flipped-side", -> collision
    trail_condition7 = plot_trail(trails[7], left_or_right_ahead = 'left') # "43-37-flipped-side",
    trail_condition11 = plot_trail(trails[11], left_or_right_ahead = 'left') #"37-43-flipped-side",
    trail_condition13 = plot_trail(trails[13], left_or_right_ahead = 'left') #"34-46-flipped-side"

    # trail_condition = plot_trail(trails[4]) #random-equal-40
    # trail_condition = plot_trail(trails[9]) #"random-equal-40-flipped-side",


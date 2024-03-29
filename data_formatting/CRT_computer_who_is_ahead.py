import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import os
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d

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

    if condition == '50-50':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 305)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 305))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '55-45':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 305)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 305))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '60-40':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 290)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 290))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


# if __name__ == '__main__':
def compute_crt(path_to_data_csv, condition):
    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
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

    # ------------------------------------------------------------------------#
    # velocity
    velocity_vehicle1 = vehicle_velocity(3, data)
    velocity_vehicle2 = vehicle_velocity(6, data)

    for i in range(len(velocity_vehicle1)):
        data_dict['velocity_vehicle1'].append(velocity_vehicle1)
        data_dict['velocity_vehicle2'].append(velocity_vehicle2)

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

    who_is_first_tunnel = min(index_of_tunnel_vehicle1, index_of_tunnel_vehicle2)
    who_is_first_tunnel_time = time_in_seconds_trail[who_is_first_tunnel]

    ##compute crt
    crt_object = calculate_conflict_resolved_time(data_dict, simulation_constants, condition)

    if not crt_object[1].size:
        crt = 0
    else:
        crt = crt_object[1][-1] - who_is_first_tunnel_time

    return crt


if __name__ == '__main__':
    # condition50-50
    files_directory1 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    condition = '50-50'

    trails_condition_50_50 = []
    for file in Path(files_directory1).glob('*.csv'):
        trails_condition_50_50.append(file)
    trails_condition_50_50 = natsorted(trails_condition_50_50, key=str)
    trails = natsorted(trails_condition_50_50, key=str)

    crts_50_50 = []
    for i in range(len(trails_condition_50_50)):
        crt = compute_crt(trails_condition_50_50[i], condition)
        crts_50_50.append(crt)

    # condition55-45 - right ahead
    files_directory2 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    condition = '55-45'
    trails_condition_45_55 = []
    for file in Path(files_directory2).glob('*.csv'):
        trails_condition_45_55.append(file)
    trails_condition_45_55 = natsorted(trails_condition_45_55, key=str)
    trails = natsorted(trails_condition_45_55, key=str)

    crts_45_55 = []
    for i in range(len(trails_condition_45_55)):
        crt = compute_crt(trails_condition_45_55[i], condition)
        crts_45_55.append(crt)

    # condition55-45 - left ahead
    files_directory3 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    condition = '55-45'
    trails_condition_55_45 = []
    for file in Path(files_directory3).glob('*.csv'):
        trails_condition_55_45.append(file)
    trails_condition_55_45 = natsorted(trails_condition_55_45, key=str)

    crts_55_45 = []
    # crt_indexes_55_45 = []
    for i in range(len(trails_condition_55_45)):
        crt = compute_crt(trails_condition_55_45[i], condition)
        crts_55_45.append(crt)

    # condition right ahead
    files_directory4 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'
    trails_condition_40_60 = []
    condition = '60-40'
    for file in Path(files_directory4).glob('*.csv'):
        trails_condition_40_60.append(file)
    trails_condition_40_60 = natsorted(trails_condition_40_60, key=str)

    crts_40_60 = []
    for i in range(len(trails_condition_40_60)):
        crt = compute_crt(trails_condition_40_60[i], condition)
        crts_40_60.append(crt)

    # condition left ahead
    files_directory5 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'
    trails_condition_60_40 = []
    condition = '60-40'
    for file in Path(files_directory5).glob('*.csv'):
        trails_condition_60_40.append(file)
    trails_condition_60_40 = natsorted(trails_condition_60_40, key=str)

    crts_60_40 = []
    # crt_indexes_60_40 = []
    for i in range(len(trails_condition_60_40)):
        crt = compute_crt(trails_condition_60_40[i], condition)
        crts_60_40.append(crt)

    path_to_saved_dict_crt = os.path.join('..', 'data_folder', 'crt_who_is_ahead.csv')

    df1 = pd.DataFrame({'crt_50_50': crts_50_50})
    df2 = pd.DataFrame({'crt_45_55_vehicle1': crts_45_55})
    df3 = pd.DataFrame({'crt_55_45_vehicle2': crts_55_45})
    df4 = pd.DataFrame({'crt_40_60_vehicle1': crts_40_60})
    df5 = pd.DataFrame({'crt_60_40_vehicle2': crts_60_40})
    pd.concat([df1, df2, df3, df4, df5], axis=1).to_csv(path_to_saved_dict_crt, index=False)

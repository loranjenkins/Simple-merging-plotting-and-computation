import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path
import seaborn as sns
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
        list_y.append(y_loc)

    return list_x, list_y


def vehicle_velocity(intcolumnname, data_csv):
    velocity = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        velocity_vehicle = eval(data_csv.iloc[i, intcolumnname])
        x_loc = velocity_vehicle[0]
        velocity.append(x_loc)
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

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


# if __name__ == '__main__':
def get_dict(path_to_data_csv, who_is_ahead, condition):
    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=135,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)

    files_directory = path_to_data_csv
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    all_pds_list = []
    for i in range(len(trails)):
        all_pds = pd.read_csv(trails[i], sep=',')
        all_pds_list.append(all_pds)

    xy_coordinates_for_vehicle1 = []
    xy_coordinates_for_vehicle2 = []

    for i in range(len(all_pds_list)):
        xy_coordinates_vehicle = vehicle_xy_coordinates(2, all_pds_list[i])
        xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, all_pds_list[i])

        xy_coordinates_zip = [list(a) for a in zip(xy_coordinates_vehicle[0], xy_coordinates_vehicle[1])]
        xy_coordinates_zip2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

        xy_coordinates_for_vehicle1.append(xy_coordinates_zip)
        xy_coordinates_for_vehicle2.append(xy_coordinates_zip2)

    # compute travelled distance for indexes tunnel and merge
    travelled_distance_vehicle1 = []
    travelled_distance_vehicle2 = []
    for list_index in range(len(xy_coordinates_for_vehicle1)):
        individual_xy_vehicle1 = xy_coordinates_for_vehicle1[list_index]
        individual_xy_vehicle2 = xy_coordinates_for_vehicle2[list_index]
        inner_xy_list_1 = []
        inner_xy_list_2 = []
        for xy in range(len(individual_xy_vehicle1)):
            traveled_distance_vehicle_1 = track.coordinates_to_traveled_distance(individual_xy_vehicle1[xy])
            traveled_distance_vehicle_2 = track.coordinates_to_traveled_distance(individual_xy_vehicle2[xy])
            inner_xy_list_1.append(traveled_distance_vehicle_1)
            inner_xy_list_2.append(traveled_distance_vehicle_2)

        travelled_distance_vehicle1.append(inner_xy_list_1)
        travelled_distance_vehicle2.append(inner_xy_list_2)

    # compute the indexes
    indexes_of_tunnel_vehicle1 = []
    indexes_of_tunnel_vehicle2 = []
    for list_index in range(len(travelled_distance_vehicle1)):
        individual_trace_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_trace_vehicle2 = travelled_distance_vehicle2[list_index]

        index_of_tunnel_vehicle_1 = min(range(len(individual_trace_vehicle1)), key=lambda i: abs(
            individual_trace_vehicle1[i] - (track.tunnel_length-10)))

        index_of_tunnel_vehicle_2 = min(range(len(individual_trace_vehicle2)), key=lambda i: abs(
            individual_trace_vehicle2[i] - (track.tunnel_length-10)))

        indexes_of_tunnel_vehicle1.append(index_of_tunnel_vehicle_1)
        indexes_of_tunnel_vehicle2.append(index_of_tunnel_vehicle_2)

    dict_for_spag = {'trial': [], 'time_ahead': [], 'velocity_ahead': []}

    velocity_list = []
    time = []
    if condition == '60-40':
        if who_is_ahead == 'vehicle1':
            for pandas in range(len(all_pds_list)):
                time_in_datetime = get_timestamps(0, all_pds_list[pandas])
                time_in_seconds_trail_ = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]

                index_of_other_vehicle_leaving = min(range(len(time_in_seconds_trail_)), key=lambda i: abs(
                    time_in_seconds_trail_[i] - (time_in_seconds_trail_[indexes_of_tunnel_vehicle2[pandas]])))

                time_in_seconds_trail = time_in_seconds_trail_[
                                        indexes_of_tunnel_vehicle1[pandas]:index_of_other_vehicle_leaving]

                time.append(time_in_seconds_trail)

                velocity_vehicle = vehicle_velocity(3, all_pds_list[pandas])
                velocity_vehicle = velocity_vehicle[indexes_of_tunnel_vehicle1[pandas]:index_of_other_vehicle_leaving]
                velocity_list.append(velocity_vehicle)
        elif who_is_ahead == 'vehicle2':
            for pandas in range(len(all_pds_list)):
                time_in_datetime = get_timestamps(0, all_pds_list[pandas])
                time_in_seconds_trail_ = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]

                index_of_other_vehicle_leaving = min(range(len(time_in_seconds_trail_)), key=lambda i: abs(
                    time_in_seconds_trail_[i] - (time_in_seconds_trail_[indexes_of_tunnel_vehicle1[pandas]])))

                time_in_seconds_trail = time_in_seconds_trail_[
                                        indexes_of_tunnel_vehicle2[pandas]:index_of_other_vehicle_leaving]

                time.append(time_in_seconds_trail)

                velocity_vehicle = vehicle_velocity(6, all_pds_list[pandas])
                velocity_vehicle = velocity_vehicle[indexes_of_tunnel_vehicle2[pandas]:index_of_other_vehicle_leaving]
                velocity_list.append(velocity_vehicle)

    if condition == '55-45':
        if who_is_ahead == 'vehicle1':
            for pandas in range(len(all_pds_list)):
                time_in_datetime = get_timestamps(0, all_pds_list[pandas])
                time_in_seconds_trail_ = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]

                index_of_other_vehicle_leaving = min(range(len(time_in_seconds_trail_)), key=lambda i: abs(
                    time_in_seconds_trail_[i] - (time_in_seconds_trail_[indexes_of_tunnel_vehicle2[pandas]])))

                time_in_seconds_trail = time_in_seconds_trail_[
                                        indexes_of_tunnel_vehicle1[pandas]:index_of_other_vehicle_leaving]

                time.append(time_in_seconds_trail)

                velocity_vehicle = vehicle_velocity(3, all_pds_list[pandas])
                velocity_vehicle = velocity_vehicle[indexes_of_tunnel_vehicle1[pandas]:index_of_other_vehicle_leaving]
                velocity_list.append(velocity_vehicle)
        elif who_is_ahead == 'vehicle2':
            for pandas in range(len(all_pds_list)):
                time_in_datetime = get_timestamps(0, all_pds_list[pandas])
                time_in_seconds_trail_ = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]

                index_of_other_vehicle_leaving = min(range(len(time_in_seconds_trail_)), key=lambda i: abs(
                    time_in_seconds_trail_[i] - (time_in_seconds_trail_[indexes_of_tunnel_vehicle1[pandas]])))

                time_in_seconds_trail = time_in_seconds_trail_[
                                        indexes_of_tunnel_vehicle2[pandas]:index_of_other_vehicle_leaving]

                time.append(time_in_seconds_trail)

                velocity_vehicle = vehicle_velocity(6, all_pds_list[pandas])
                velocity_vehicle = velocity_vehicle[indexes_of_tunnel_vehicle2[pandas]:index_of_other_vehicle_leaving]
                velocity_list.append(velocity_vehicle)

    trial = 1
    for list_in in range(len(velocity_list)):
        inner = velocity_list[list_in]
        for i in range(len(inner)):
            dict_for_spag['velocity_ahead'].append(inner[i])
            dict_for_spag['trial'].append(trial)
        trial += 1

    for list_in in range(len(time)):
        inner = time[list_in]
        for i in range(len(inner)):
            dict_for_spag['time_ahead'].append(inner[i])

    return dict_for_spag


if __name__ == '__main__':
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'
    [path_to_csv_vehicle1_ahead,path_to_csv_vehicle1_ahead]

    who_is_ahead = 'vehicle1'
    condition = '60-40'
    dict60_40_v1_ahead = get_dict(path_to_csv_vehicle1_ahead, who_is_ahead, condition)

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'

    who_is_ahead = 'vehicle2'
    condition = '60-40'
    dict60_40_v2_ahead = get_dict(path_to_csv_vehicle2_ahead, who_is_ahead, condition)

    df1 = pd.DataFrame.from_dict(dict60_40_v1_ahead)
    df2 = pd.DataFrame.from_dict(dict60_40_v2_ahead)
    df2['trial'] = df2['trial'] + list(df1['trial'])[-1]
    df_60_40 = pd.concat([df1, df2])


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Velocity traces ahead vehicle exit tunnel behind vehicle\n condition 3 (60-40 km/h)')
    sns.lineplot(data=df_60_40, x=df_60_40['time_ahead'], y=df_60_40['velocity_ahead'], hue=df_60_40['trial'], ax= ax)
    ax.get_legend().remove()
    ax.set(xlabel='Time [s]', ylabel='Velocity [km/h]')


    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    who_is_ahead = 'vehicle1'
    condition = '55-45'
    dict55_45_v1_ahead = get_dict(path_to_csv_vehicle1_ahead, who_is_ahead, condition)

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    who_is_ahead = 'vehicle2'
    condition = '55-45'
    dict55_45_v2_ahead = get_dict(path_to_csv_vehicle2_ahead, who_is_ahead, condition)

    df3 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df4 = pd.DataFrame.from_dict(dict55_45_v2_ahead)
    df4['trial'] = df4['trial'] + list(df3['trial'])[-1]

    df_55_45 = pd.concat([df3, df4])
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Velocity traces ahead vehicle exit tunnel behind vehicle\n condition 2 (55-45 km/h)')
    sns.lineplot(data=df_55_45, x=df_55_45['time_ahead'], y=df_55_45['velocity_ahead'], hue=df_55_45['trial'], ax=ax1)
    ax1.get_legend().remove()
    ax1.set(xlabel='Time [s]', ylabel='Velocity [km/h]')

    plt.show()


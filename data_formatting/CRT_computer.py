import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import os
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

    approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                     (np.array(data_dict['distance_traveled_vehicle1']) < track.section_length_before)) | \
                    ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                     (np.array(data_dict['distance_traveled_vehicle2']) < track.section_length_before))
    indices_of_conflict_resolved = (on_collision_course & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


# if __name__ == '__main__':
def compute_crt(path_to_data_csv):

    data = pd.read_csv(path_to_data_csv, sep=',')

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

    data_dict = {'time': [],
                 'velocity_vehicle1': [],
                 'velocity_vehicle2': [],
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

    ##compute crt
    crt_object = calculate_conflict_resolved_time(data_dict, simulation_constants)

    if not crt_object[1].size:
        crt = 0
    else:
        crt = crt_object[1][-1]

    if not crt_object[1].size:
        crt_index = 0
    else:
        crt_index = min(range(len(data_dict['time'])),
                                   key=lambda i: abs(data_dict['time'][i] - crt))

    return crt, crt_index

if __name__ == '__main__':
    #condition50-50
    files_directory1 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50'

    trails_condition_50_50 = []
    for file in Path(files_directory1).glob('*.csv'):
        trails_condition_50_50.append(file)
    trails_condition_50_50 = natsorted(trails_condition_50_50, key=str)

    crts_50_50 = []
    crt_indexes_50_50 = []
    for i in range(len(trails_condition_50_50)):
        crt = compute_crt(trails_condition_50_50[i])[0]
        crt_index = compute_crt(trails_condition_50_50[i])[1]
        crts_50_50.append(crt)
        crt_indexes_50_50.append(crt_index)

    #condition55-45
    files_directory2 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_55_45'
    trails_condition_55_45 = []
    for file in Path(files_directory2).glob('*.csv'):
        trails_condition_55_45.append(file)
    trails_condition_55_45 = natsorted(trails_condition_55_45, key=str)

    crts_55_45 = []
    crt_indexes_55_45 = []
    for i in range(len(trails_condition_55_45)):
        crt = compute_crt(trails_condition_55_45[i])[0]
        crt_index = compute_crt(trails_condition_55_45[i])[1]
        crts_55_45.append(crt)
        crt_indexes_55_45.append(crt_index)

    # condition60-40
    files_directory3 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40'
    trails_condition_60_40 = []
    for file in Path(files_directory3).glob('*.csv'):
        trails_condition_60_40.append(file)
    trails_condition_60_40 = natsorted(trails_condition_60_40, key=str)

    crts_60_40 = []
    crt_indexes_60_40 = []
    for i in range(len(trails_condition_60_40)):
        crt = compute_crt(trails_condition_60_40[i])[0]
        crt_index = compute_crt(trails_condition_60_40[i])[1]
        crts_60_40.append(crt)
        crt_indexes_60_40.append(crt_index)

    path_to_saved_dict_crt = os.path.join('..', 'data_folder', 'crt_all_conditions.csv')
    path_to_saved_dict_index = os.path.join('..', 'data_folder', 'crt_index_all_conditions.csv')


    df1 = pd.DataFrame({'crt_50_50': crts_50_50})
    df2 = pd.DataFrame({'crt_55_45': crts_55_45})
    df3 = pd.DataFrame({'crt_60_40': crts_60_40})
    pd.concat([df1, df2, df3], axis=1).to_csv(path_to_saved_dict_crt, index=False)

    df3 = pd.DataFrame({'crt_index_50_50': crt_indexes_50_50})
    df4 = pd.DataFrame({'crt_index_55_45': crt_indexes_55_45})
    df5 = pd.DataFrame({'crt_index_60_40': crt_indexes_60_40})
    pd.concat([df3, df4, df5], axis=1).to_csv(path_to_saved_dict_index, index=False)
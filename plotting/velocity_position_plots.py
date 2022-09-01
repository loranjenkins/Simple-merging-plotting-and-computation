import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
# import seaborn as sns
import datetime

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack

if __name__ == '__main__':
    # data = pd.read_csv(
    #     r'D:\Pycharmprojects\Simple-merging-plotting-and-computation\data_folder\joan_data_experment_2vehicles.csv',
    #     sep=';', converters={'Carla Interface.agents.Ego Vehicle_1.transform': literal_eval})
    data = pd.read_csv(
        r'C:\Users\localadmin\PycharmProjects\Simple-merging-plotting-and-computation\data_folder\joan_data_experment_2vehicles.csv',
        sep=';')
    data.values.tolist()
    data = data.iloc[500:,:]

def vehicle_xy_coordinates(intcolumnname):
    list_x = []
    list_y = []
    for i in range(len(data.iloc[:, intcolumnname])):
        transform_vehicle1 = eval(data.iloc[i, intcolumnname])
        x_loc = transform_vehicle1[0]
        y_loc = transform_vehicle1[1]

        list_x.append(x_loc)
        # list_x = list(dict.fromkeys(list_x))
        list_y.append(y_loc)
        # list_y = list(dict.fromkeys(list_y))

    return list_x, list_y


xy_coordinates_vehicle1 = vehicle_xy_coordinates(2)
xy_coordinates_vehicle2 = vehicle_xy_coordinates(5)

xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)

# x1, y1 = xy_coordinates_vehicle1.T
# x2, y2 = xy_coordinates_vehicle2.T
simulation_constants = SimulationConstants(vehicle_width=2,
                                           vehicle_length=4.7,
                                           track_width=8,
                                           track_height=292.5,
                                           track_start_point_distance=60, #distance between centerpoints cars -> differs each trail
                                           track_section_length_before= 294, #length of section differs each trail
                                           track_section_length_after = 150)

track = SymmetricMergingTrack(simulation_constants)
# print(xy_coordinates_vehicle1[1])
# print(track.closest_point_on_route(xy_coordinates_vehicle1[3903]))
data_dict = {'time': [],
             'x1_straight': [],
             'y1_straight': [],
             'x2_straight': [],
             'y2_straight': [],
             'velocity_vehicle1': [],
             'velocity_vehicle2': [],
             'distance_traveled_vehicle1': [],
             'distance_traveled_vehicle2': []}

for i in range(len(xy_coordinates_vehicle1)):
    straight_line = track.closest_point_on_route_rightvehicle(xy_coordinates_vehicle1[i])
    data_dict['x1_straight'].append(straight_line[0][0]+5)
    data_dict['y1_straight'].append(straight_line[0][1])

for i in range(len(xy_coordinates_vehicle2)):
    straight_line = track.closest_point_on_route_leftvehicle(xy_coordinates_vehicle2[i])
    data_dict['x2_straight'].append(straight_line[0][0]-5)
    data_dict['y2_straight'].append(straight_line[0][1])

fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('-')
# plt.title('positions over time')
ax1.set_xlabel('y position [m]')
ax1.set_ylabel('x position [m]')
ax1.set_yticks([-30, -5, 5, 30])
ax1.set_yticklabels([30, 0, 0, -30])
ax1.invert_xaxis()
# ax1.tick_params(axis='y', labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)


ax1.scatter(data_dict['y1_straight'][0::300], data_dict['x1_straight'][0::300], s=10)
ax1.scatter(data_dict['y2_straight'][0::300], data_dict['x2_straight'][0::300], s=10)

ax1.plot(data_dict['y1_straight'], data_dict['x1_straight'])
ax1.plot(data_dict['y2_straight'], data_dict['x2_straight'])


# velocity_time plot
def vehicle_velocity(intcolumnname):
    velocity = []
    for i in range(len(data.iloc[:, intcolumnname])):
        velocity_vehicle = eval(data.iloc[i, intcolumnname])
        x_loc = velocity_vehicle[0]
        velocity.append(x_loc)
        # velocity = list(dict.fromkeys(velocity))
    return velocity


def get_timestamps(intcolumnname):
    time = []
    for i in range(len(data.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        # we need to remove doubles??
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time


velocity_vehicle1 = vehicle_velocity(3)
velocity_vehicle2 = vehicle_velocity(6)

for i in range(len(velocity_vehicle1)):
    data_dict['velocity_vehicle1'].append(velocity_vehicle1)
    data_dict['velocity_vehicle2'].append(velocity_vehicle2)


time_in_datetime = get_timestamps(0)
time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
time_in_seconds_trail = np.array(time_in_seconds_trail)
data_dict['time'] = time_in_seconds_trail

# plt.xlabel('time [s]')
# plt.ylabel('velocity [m/s]')
# plt.title('velocity at times')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Velocity [m/s]')
ax2.plot(time_in_seconds_trail, velocity_vehicle1)
ax2.plot(time_in_seconds_trail, velocity_vehicle2)

fig.tight_layout(pad=1.0)

##collision bound plot

for i in range(len(xy_coordinates_vehicle1)):
    traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
    traveled_distance2 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle2[i])
    data_dict['distance_traveled_vehicle1'].append(traveled_distance1)
    data_dict['distance_traveled_vehicle2'].append(traveled_distance2)

average_travelled_distance_trace = list((np.array(data_dict['distance_traveled_vehicle1']) + np.array(
    data_dict['distance_traveled_vehicle2'])) / 2.)

headway = list(np.array(data_dict['distance_traveled_vehicle1']) - np.array(data_dict['distance_traveled_vehicle2']))

ax3.plot(average_travelled_distance_trace[::30], headway[::30])
ax3.set_xlabel('Average travelled distance [m]')
ax3.set_ylabel('Headway [m]')
fig.tight_layout(pad=1.0)

plt.show()

##compute crt

def check_if_on_collision_course_for_point(travelled_distance_collision_point, data_dict, simulation_constants):
    track = SymmetricMergingTrack(simulation_constants)
    point_predictions = {'vehicle1': [], 'vehicle2': []}

    point_predictions['vehicle1'] = np.array(data_dict['distance_traveled_vehicle1']) + np.array(data_dict['velocity_vehicle1']) * (
                travelled_distance_collision_point - np.array(data_dict['distance_traveled_vehicle2'])) / np.array(data_dict['velocity_vehicle2'])
    point_predictions['vehicle2'] = np.array(data_dict['distance_traveled_vehicle2']) + np.array(data_dict['velocity_vehicle2']) * (
            travelled_distance_collision_point - np.array(data_dict['distance_traveled_vehicle1'])) / np.array(data_dict['velocity_vehicle1'])

    # lb, ub = track.get_collision_bounds(travelled_distance_collision_point, simulation_constants.vehicle_width, simulation_constants.vehicle_length)
    lb = 150
    ub = 0
    on_collision_course = ((lb < point_predictions['vehicle2']) & (point_predictions['vehicle2'] < ub)) | \
                          ((lb < point_predictions['vehicle1']) & (point_predictions['vehicle1'] < ub))
    return on_collision_course

def calculate_conflict_resolved_time(data_dict, simulation_constants):
    time = data_dict['time']
    track = SymmetricMergingTrack(simulation_constants)

    merge_point_collision_course = check_if_on_collision_course_for_point(2 * track._section_length_before, data_dict, simulation_constants)
    threshold_collision_course = check_if_on_collision_course_for_point(track._upper_bound_threshold + 1e-3, data_dict, simulation_constants)

    on_collision_course = merge_point_collision_course | threshold_collision_course

    approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track._section_length_before) &
                     (np.array(data_dict['distance_traveled_vehicle1']) < 2 * track._section_length_before)) | \
                    ((np.array(data_dict['distance_traveled_vehicle2']) > track._section_length_before) &
                     (np.array(data_dict['distance_traveled_vehicle2']) < 2 * track._section_length_before))

    indices_of_conflict_resolved = ((on_collision_course == False) & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved][0]
    except IndexError:
        time_of_conflict_resolved = None

    return time_of_conflict_resolved

# crt = calculate_conflict_resolved_time(data_dict, simulation_constants)
# print(crt)
#
# lb, ub = track.get_collision_bounds(2 * track._section_length_before, simulation_constants.vehicle_width, simulation_constants.vehicle_length)
# print(lb, ub)
# merge_point_collision_course = check_if_on_collision_course_for_point((track._section_length_before+track._section_length_after), data_dict, simulation_constants)
# print(list(merge_point_collision_course))
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
# import seaborn as sns
import pickle
import datetime

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack

if __name__ == '__main__':
    data = pd.read_csv(
        'C:\\Users\loran\Desktop\Data_CRT\joan_data_20220901_14h00m40s.csv',
        sep=';')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:,:]
    data = data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=150)

    track = SymmetricMergingTrack(simulation_constants)

def vehicle_xy_coordinates(intcolumn):
    list_x = []
    list_y = []
    for i in range(len(data.iloc[:, intcolumn])):
        transform_vehicle1 = eval(data.iloc[i, intcolumn])
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
    straight_line_vehicle1 = track.closest_point_on_route(xy_coordinates_vehicle1[i])
    data_dict['x1_straight'].append(straight_line_vehicle1[0][0]+20)
    data_dict['y1_straight'].append(straight_line_vehicle1[0][1])


for i in range(len(xy_coordinates_vehicle2)):
    straight_line_vehicle2 = track.closest_point_on_route(xy_coordinates_vehicle2[i])
    data_dict['x2_straight'].append(straight_line_vehicle2[0][0]-20)
    data_dict['y2_straight'].append(straight_line_vehicle2[0][1])

fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('-')
# plt.title('positions over time')
ax1.set_xlabel('y position [m]')
ax1.set_ylabel('x position [m]')
ax1.set_yticks([175, -20, 20, -175])
ax1.set_yticklabels([175, 0, 0, -175])


# ax1.tick_params(axis='y', labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)

ax1.scatter(data_dict['y1_straight'][0::300], data_dict['x1_straight'][0::300], s=10)
ax1.scatter(data_dict['y2_straight'][0::300], data_dict['x2_straight'][0::300], s=10)
ax1.set_xlim(50,450)

ax1.plot(data_dict['y1_straight'], data_dict['x1_straight'])
ax1.plot(data_dict['y2_straight'], data_dict['x2_straight'])
# ax1.plot(data_dict['x1_straight'], data_dict['y1_straight'])
# ax1.plot(data_dict['x2_straight'], data_dict['y2_straight'])




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
ax2.set_xlim(0, 30)
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

headway = np.array(data_dict['distance_traveled_vehicle1']) - np.array(data_dict['distance_traveled_vehicle2'])

ax3.plot(average_travelled_distance_trace, headway)
ax3.set_xlabel('Average travelled distance [m]')
ax3.set_ylabel('Headway [m]')
ax3.set_xlim(50,450)
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
    #what are these point_predictions -> start array with section_length_before

    lb, ub = track.get_collision_bounds(travelled_distance_collision_point, simulation_constants.vehicle_width, simulation_constants.vehicle_length)
    #

    on_collision_course = ((lb < point_predictions['vehicle2']) & (point_predictions['vehicle2'] < ub)) | \
                          ((lb < point_predictions['vehicle1']) & (point_predictions['vehicle1'] < ub))

    return on_collision_course

def calculate_conflict_resolved_time(data_dict, simulation_constants):
    time = data_dict['time']
    track = SymmetricMergingTrack(simulation_constants)

    merge_point_collision_course = check_if_on_collision_course_for_point(track.section_length_before, data_dict,
                                                                          simulation_constants)

    threshold_collision_course = check_if_on_collision_course_for_point(track.upper_bound_threshold + 1e-3, data_dict, simulation_constants)
    # 1e-3 is used for straight approach (always inside)
    end_point_collision_course = check_if_on_collision_course_for_point(track.section_length_before + track.section_length_after,
                                                                        data_dict, simulation_constants)

    on_collision_course = merge_point_collision_course \
                          | threshold_collision_course \
                          | end_point_collision_course


    approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > 120) & # tunnel length
                     (np.array(data_dict['distance_traveled_vehicle1']) < track.section_length_before)) | \
                    ((np.array(data_dict['distance_traveled_vehicle2']) > 120) &
                     (np.array(data_dict['distance_traveled_vehicle2']) < track.section_length_before))
    # why times 2 here

    indices_of_conflict_resolved = ((on_collision_course == False) & approach_mask)

    # print(np.array[indices_of_conflict_resolved][0]) # this one not working

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved][0]
    except IndexError:
        time_of_conflict_resolved = None

    return time_of_conflict_resolved


crt = calculate_conflict_resolved_time(data_dict, simulation_constants)
print(crt)

a_file = open("global_data_dict.pkl", "wb")
pickle.dump(data_dict, a_file)
a_file.close()


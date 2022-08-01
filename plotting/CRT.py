import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
import datetime

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack
# from trackobjects.trackside import TrackSide


if __name__ == '__main__':

    data = pd.read_csv(r'D:\Pycharmprojects\Simple-merging-plotting-and-computation\data_folder\joan_data_experment_2vehicles.csv', sep = ';', converters = {'Carla Interface.agents.Ego Vehicle_1.transform': literal_eval})
    data.values.tolist()

def vehicle_xy_coordinates(intcolumnname):
    list_x = []
    list_y = []
    for i in range(len(data.iloc[:, intcolumnname])):
        transform_vehicle1 = literal_eval(data.iloc[i, intcolumnname])
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
simulation_constants = SimulationConstants(vehicle_width=1.8,
                                           vehicle_length=4.5,
                                           track_start_point_distance=60,
                                           track_section_length=294.)

track = SymmetricMergingTrack(simulation_constants)
# print(xy_coordinates_vehicle1[1])
# print(track.closest_point_on_route(xy_coordinates_vehicle1[3903]))
data_dict = {'x1_straight': [],
             'y1_straight': [],
             'x2_straight': [],
             'y2_straight': [],
             'distance_traveled_vehicle1': [],
             'distance_traveled_vehicle2': []}

for i in range(len(xy_coordinates_vehicle1)):
    straight_line = track.closest_point_on_route(xy_coordinates_vehicle1[i])
    data_dict['x1_straight'].append(straight_line[0][0])
    data_dict['y1_straight'].append(straight_line[0][1])

for i in range(len(xy_coordinates_vehicle2)):
    straight_line = track.closest_point_on_route(xy_coordinates_vehicle2[i])
    data_dict['x2_straight'].append(straight_line[0][0])
    data_dict['y2_straight'].append(straight_line[0][1])

xy_coordinates_vehicle1 = [list(a) for a in zip(data_dict['x1_straight'], data_dict['y1_straight'])]
xy_coordinates_vehicle2 = [list(a) for a in zip(data_dict['x2_straight'], data_dict['y2_straight'])]

# print(xy_coordinates_vehicle2)

for i in range(len(xy_coordinates_vehicle1)):
    traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
    traveled_distance2 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle2[i])
    data_dict['distance_traveled_vehicle1'].append(traveled_distance1)
    data_dict['distance_traveled_vehicle2'].append(traveled_distance2)

print(data_dict['distance_traveled_vehicle1'])
print(data_dict['distance_traveled_vehicle2'])

#



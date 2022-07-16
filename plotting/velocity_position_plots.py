import pandas as pd
import numpy as np
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
import time

from trackobjects.trackside import TrackSide

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
        list_x = list(dict.fromkeys(list_x))
        list_y.append(y_loc)
        list_y = list(dict.fromkeys(list_y))

    return list_x, list_y

xy_coordinates_vehicle1 = vehicle_xy_coordinates(2)
xy_coordinates_vehicle2 = vehicle_xy_coordinates(5)

xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
xy_coordinates_vehicle2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

xy_coordinates_vehicle1 = np.array(xy_coordinates_vehicle1)
xy_coordinates_vehicle2 = np.array(xy_coordinates_vehicle2)

x1, y1 = xy_coordinates_vehicle1.T
x2, y2 = xy_coordinates_vehicle2.T

# plt.plot(x1, y1) --> we need closest point on route to draw straight line! ask this
# plt.plot(x2, y2)
# plt.show()

# velocity_time plot
def vehicle_velocity(intcolumnname):
    velocity = []
    for i in range(len(data.iloc[:, intcolumnname])):
        velocity_vehicle = literal_eval(data.iloc[i, intcolumnname])
        x_loc = velocity_vehicle[0]
        velocity.append(x_loc)
        # velocity = list(dict.fromkeys(velocity))
    return velocity

def get_timestamps(intcolumnname):
    time = []
    for i in range(len(data.iloc[:, intcolumnname])):
       time_s = data.iloc[i, intcolumnname]
       time.append(time_s)
    return time

velocity_vehicle1 = vehicle_velocity(3)
velocity_vehicle2 = vehicle_velocity(6)
time = get_timestamps(0)

plt.plot(time, velocity_vehicle1)
plt.plot(time, velocity_vehicle2)
plt.show()




    # plt.plot(data['travelled_distance'][side], data['velocities'][side], label=str(side))
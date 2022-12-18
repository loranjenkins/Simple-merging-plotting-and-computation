import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


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
def compute_hmd_rots(path_to_data_csv):

    data = pd.read_csv(path_to_data_csv, sep=',')

    data.drop(data.loc[data['Carla Interface.time'] == 0].index, inplace=True)
    data = data.iloc[10:, :]
    data.drop_duplicates(subset=['Carla Interface.time'], keep=False)

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=125,
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
    hmd_rots_total = v1 + v2

    return hmd_rots_total

if __name__ == '__main__':
    files_directory = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50 - kopie'
    files_directory1 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50 - kopie'
    files_directory2 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50 - kopie'

    trails = []
    for file in Path(files_directory).glob('*.csv'):
        trails.append(file)

    for file in Path(files_directory1).glob('*.csv'):
        trails.append(file)

    for file in Path(files_directory2).glob('*.csv'):
        trails.append(file)

    nested_list = []
    for i in range(len(trails)):
        innerlist = compute_hmd_rots(trails[i])
        nested_list.append(innerlist)

    resultList = []

    # Traversing in till the length of the input list of lists
    for m in range(len(nested_list)):

       # using nested for loop, traversing the inner lists
       for n in range(len(nested_list[m])):

          # Add each element to the result list
          resultList.append(nested_list[m][n])

    sns.histplot(resultList)

    plt.show()

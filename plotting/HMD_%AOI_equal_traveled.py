import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import numpy as np
import seaborn as sns
from scipy import interpolate
import os
import numpy as np

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


def interpolate(inp, fi):
    i, f = int(fi // 1), fi % 1  # Split floating-point index into whole & fractional parts.
    j = i + 1 if f > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


def average(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))


def plot_varjo(path_to_csv_folder):
    # simulation_constants = SimulationConstants(vehicle_width=2,
    #                                            vehicle_length=4.7,
    #                                            tunnel_length=118,  # original = 118 -> check in unreal
    #                                            track_width=8,
    #                                            track_height=215,
    #                                            track_start_point_distance=430,
    #                                            track_section_length_before=304.056,
    #                                            track_section_length_after=150)  # goes until 400

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=135,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)

    files_directory = path_to_csv_folder
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)

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
    indexes_of_tunnel_and_merge_vehicle1 = []
    indexes_of_tunnel_and_merge_vehicle2 = []
    for list_index in range(len(travelled_distance_vehicle1)):
        individual_trace_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_trace_vehicle2 = travelled_distance_vehicle2[list_index]
        inner_index_list_1 = []
        inner_index_list_2 = []
        for i in range(0, 1):
            index_of_tunnel_vehicle_1 = min(range(len(individual_trace_vehicle1)), key=lambda i: abs(
                individual_trace_vehicle1[i] - track.tunnel_length))

            index_of_tunnel_vehicle_2 = min(range(len(individual_trace_vehicle2)), key=lambda i: abs(
                individual_trace_vehicle2[i] - track.tunnel_length))

            index_of_mergepoint_vehicle1 = min(range(len(individual_trace_vehicle1)),
                                               key=lambda i: abs(
                                                   individual_trace_vehicle1[i] - track.section_length_before))
            index_of_mergepoint_vehicle2 = min(range(len(individual_trace_vehicle2)),
                                               key=lambda i: abs(
                                                   individual_trace_vehicle2[i] - track.section_length_before))

            inner_index_list_1.append(index_of_tunnel_vehicle_1)
            inner_index_list_1.append(index_of_mergepoint_vehicle1)

            inner_index_list_2.append(index_of_tunnel_vehicle_2)
            inner_index_list_2.append(index_of_mergepoint_vehicle2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_index_list_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_index_list_2)

    # interactive data for each vehicle
    interactive_area_travelled_trace_vehicle1 = []
    hmd_rot_interactive_area_vehicle1 = []

    interactive_area_travelled_trace_vehicle2 = []
    hmd_rot_interactive_area_vehicle2 = []

    for i in range(len(indexes_of_tunnel_and_merge_vehicle1)):
        hmd_rot_1 = list(all_pds_list[i]['HMD_rotation_vehicle1'][
                         indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]])

        hmd_rot_2 = list(all_pds_list[i]['HMD_rotation_vehicle2'][
                         indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]])

        interactive_trace_1 = travelled_distance_vehicle1[i][
                              indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]]
        interactive_trace_2 = travelled_distance_vehicle2[i][
                              indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]]

        hmd_rot_interactive_area_vehicle1.append(hmd_rot_1)
        hmd_rot_interactive_area_vehicle2.append(hmd_rot_2)

        interactive_area_travelled_trace_vehicle1.append(interactive_trace_1)
        interactive_area_travelled_trace_vehicle2.append(interactive_trace_2)

        on_ramp_vs_opponent_vehicle1 = []
        on_ramp_vs_opponent_vehicle2 = []

        for list_index in range(len(hmd_rot_interactive_area_vehicle1)):
            individual_hmd_rot_list_1 = hmd_rot_interactive_area_vehicle1[list_index]
            individual_hmd_rot_list_2 = hmd_rot_interactive_area_vehicle2[list_index]

            inner_attention_list_1 = []
            inner_attention_list_2 = []

            for i in range(len(individual_hmd_rot_list_1)):
                if individual_hmd_rot_list_1[i] > 0.95:  # this we need to know better
                    inner_attention_list_1.append(1)
                else:
                    inner_attention_list_1.append(0)

            for i in range(len(individual_hmd_rot_list_2)):
                if individual_hmd_rot_list_2[i] > 0.95:  # this we need to know better
                    inner_attention_list_2.append(1)
                else:
                    inner_attention_list_2.append(0)

            on_ramp_vs_opponent_vehicle1.append(inner_attention_list_1)
            on_ramp_vs_opponent_vehicle2.append(inner_attention_list_2)

    # interpolation to get equal lengths list HMD rots
    max_len_v1 = []
    max_len_v2 = []
    for i in range(len(interactive_area_travelled_trace_vehicle1)):
        a = len(interactive_area_travelled_trace_vehicle1[i])
        max_len_v1.append(a)

    for i in range(len(interactive_area_travelled_trace_vehicle2)):
        a = len(interactive_area_travelled_trace_vehicle2[i])
        max_len_v2.append(a)

    max_len_v1 = max(max_len_v1)
    max_len_v2 = max(max_len_v2)

    equal_on_ramp_vs_opponent_vehicle1 = []
    equal_on_ramp_vs_opponent_vehicle2 = []
    for index in range(len(on_ramp_vs_opponent_vehicle1)):
        delta = (len(on_ramp_vs_opponent_vehicle1[index]) - 1) / (max_len_v1 - 1)
        outp_v1 = [interpolate(on_ramp_vs_opponent_vehicle1[index], i * delta) for i in range((max_len_v1 - 1))]
        equal_on_ramp_vs_opponent_vehicle1.append(outp_v1)

    for index in range(len(on_ramp_vs_opponent_vehicle2)):
        delta = (len(on_ramp_vs_opponent_vehicle2[index]) - 1) / (max_len_v2 - 1)
        outp_v2 = [interpolate(on_ramp_vs_opponent_vehicle2[index], i * delta) for i in range(max_len_v2 - 1)]
        equal_on_ramp_vs_opponent_vehicle2.append(outp_v2)

    average_hmd_vehicle1 = list(average(equal_on_ramp_vs_opponent_vehicle1))
    average_hmd_vehicle2 = list(average(equal_on_ramp_vs_opponent_vehicle2))

    average_trace_vehicle1 = sum(average_hmd_vehicle1) / len(average_hmd_vehicle1)
    average_trace_vehicle2 = sum(average_hmd_vehicle2) / len(average_hmd_vehicle2)

    ysmoothed_1 = gaussian_filter1d(average_hmd_vehicle1, sigma=4)
    ysmoothed_2 = gaussian_filter1d(average_hmd_vehicle2, sigma=4)

    x_vehicle1 = list(np.linspace(simulation_constants.tunnel_length, simulation_constants.track_section_length_before,
                                  len(average_hmd_vehicle1)))
    x_vehicle2 = list(np.linspace(simulation_constants.tunnel_length, simulation_constants.track_section_length_before,
                                  len(average_hmd_vehicle2)))

    return ysmoothed_1, ysmoothed_2, x_vehicle1, x_vehicle2, average_trace_vehicle1, average_trace_vehicle2


if __name__ == '__main__':
    path_to_data_csv = os.path.join('..', 'data_folder', 'crt_index_first_exit.csv')
    global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

    # --------------------------------------------------
    # #left ahead 45-55
    path_to_csv_test = r'D:\Thesis_data_all_experiments\Conditions\test'

    Varjo_data = plot_varjo(path_to_csv_test)



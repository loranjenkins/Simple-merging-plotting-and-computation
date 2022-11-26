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
    path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt_index.csv')
    global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

    # --------------------------------------------------
    # #left ahead 45-55
    path_to_csv_45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\left'
    Varjo_data = plot_varjo(path_to_csv_45_55)
    # right ahead 45-55
    path_to_csv_45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\right'
    Varjo_data1 = plot_varjo(path_to_csv_45_55)

    #interpolation
    max_len_x_behind = max(len(Varjo_data[2]), len(Varjo_data1[3]))
    delta = (len(Varjo_data[2]) - 1) / (max_len_x_behind - 1)
    outp_x_behind = [interpolate(Varjo_data[2], i * delta) for i in range(max_len_x_behind)]

    max_len_hmd_behind = max(len(Varjo_data[0]), len(Varjo_data1[1]))
    delta = (len(Varjo_data[0]) - 1) / (max_len_hmd_behind - 1)
    outp_hmd_behind = [interpolate(Varjo_data[0], i * delta) for i in range(max_len_hmd_behind)]

    x_data_behind = [outp_x_behind, Varjo_data1[3]]
    hmd_data_behind = [outp_hmd_behind, Varjo_data1[1]]
    combined_x_data_behind = list(average(x_data_behind))
    combined_hmd_data_behind = list(average(hmd_data_behind))
    average_trace_behind = sum([Varjo_data[4], Varjo_data1[5]]) / 2

    #interpolation
    max_len_x_ahead = max(len(Varjo_data[3]), len(Varjo_data1[2]))
    delta = (len(Varjo_data[3]) - 1) / (max_len_x_ahead - 1)
    outp_x_ahead = [interpolate(Varjo_data[3], i * delta) for i in range(max_len_x_ahead)]

    max_len_hmd_ahead = max(len(Varjo_data[1]), len(Varjo_data1[0]))
    delta = (len(Varjo_data[1]) - 1) / (max_len_hmd_ahead - 1)
    outp_hmd_ahead = [interpolate(Varjo_data[1], i * delta) for i in range(max_len_hmd_ahead)]

    x_data_ahead = [outp_x_ahead, Varjo_data1[2]]
    hmd_data_velocity_ahead = [outp_hmd_ahead, Varjo_data1[0]]
    combined_x_data_ahead = list(average(x_data_ahead))
    combined_hmd_data_ahead = list(average(hmd_data_velocity_ahead))
    average_trace_ahead = sum([Varjo_data[5], Varjo_data1[4]]) / 2

    #plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Comparison gaze behavior over traveled distance condition 2 (55-45 km/h)')
    ax1.plot(combined_x_data_behind, combined_hmd_data_behind)  # see x below
    ax1.fill_between(combined_x_data_behind, combined_hmd_data_behind, color='blue', alpha=0.1,
                     label='Fixation on road')
    ax1.fill_between(combined_x_data_behind, combined_hmd_data_behind, 1, color='red', alpha=0.1,
                     label='Fixation on opponent')
    ax1.set_title('Participant is behind')
    ax1.set_xlim([135, 325])
    ax1.set_ylim([0, 1])

    ax2.plot(combined_x_data_ahead, combined_hmd_data_ahead)  # see x below
    ax2.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, color='blue', alpha=0.1, label='Fixation on road')
    ax2.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, 1, color='red', alpha=0.1,
                     label='Fixation on opponent')
    ax2.set_title('Participant is ahead')
    ax2.set_xlim([135, 325])
    ax2.set_ylim([0, 1])

    ax1.axvline(global_crt_index['median_55_45_v1'][0], 0, 1, color='r', label='Average conflict resolved')
    ax2.axvline(global_crt_index['median_55_45_v2'][0], 0, 1, color='r', label='Average conflict resolved')

    fig.text(0.05, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "Average traveled distance [m]", ha="center", va="center")
    ax1.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_behind, 2)))
    ax2.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_ahead, 2)))
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')

    # left ahead 60-40
    path_to_csv_40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\left'
    Varjo_data3 = plot_varjo(path_to_csv_40_60)
    path_to_csv_60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\right'
    Varjo_data4 = plot_varjo(path_to_csv_60_40)

    #interpolation
    max_len_x_behind = max(len(Varjo_data3[2]), len(Varjo_data4[3]))
    delta = (len(Varjo_data3[2]) - 1) / (max_len_x_behind - 1)
    outp_x_behind = [interpolate(Varjo_data3[2], i * delta) for i in range(max_len_x_behind)]

    max_len_hmd_behind = max(len(Varjo_data3[0]), len(Varjo_data4[1]))
    delta = (len(Varjo_data3[0]) - 1) / (max_len_hmd_behind - 1)
    outp_hmd_behind = [interpolate(Varjo_data3[0], i * delta) for i in range(max_len_hmd_behind)]

    x_data_behind = [outp_x_behind, Varjo_data4[3]]
    hmd_data_behind = [outp_hmd_behind, Varjo_data4[1]]
    combined_x_data_behind = list(average(x_data_behind))
    combined_hmd_data_behind = list(average(hmd_data_behind))
    average_trace_behind = sum([Varjo_data3[4], Varjo_data4[5]]) / 2

    #interpolation
    max_len_x_ahead = max(len(Varjo_data3[3]), len(Varjo_data4[2]))
    delta = (len(Varjo_data4[3]) - 1) / (max_len_x_ahead - 1)
    outp_x_ahead = [interpolate(Varjo_data4[3], i * delta) for i in range(max_len_x_ahead)]

    max_len_hmd_ahead = max(len(Varjo_data3[1]), len(Varjo_data4[0]))
    delta = (len(Varjo_data4[1]) - 1) / (max_len_hmd_ahead - 1)
    outp_hmd_ahead = [interpolate(Varjo_data4[1], i * delta) for i in range(max_len_hmd_ahead)]

    x_data_ahead = [Varjo_data3[3], outp_x_ahead]
    hmd_data_velocity_ahead = [Varjo_data3[1], outp_hmd_ahead]
    combined_x_data_ahead = list(average(x_data_ahead))
    combined_hmd_data_ahead = list(average(hmd_data_velocity_ahead))
    average_trace_ahead = sum([Varjo_data3[5], Varjo_data4[4]]) / 2

    #plotting
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Comparison gaze behavior over traveled distance condition 3 (60-40 km/h)')
    ax3.plot(combined_x_data_behind, combined_hmd_data_behind)  # see x below
    ax3.fill_between(combined_x_data_behind, combined_hmd_data_behind, color='blue', alpha=0.1,
                     label='Fixation on road')
    ax3.fill_between(combined_x_data_behind, combined_hmd_data_behind, 1, color='red', alpha=0.1,
                     label='Fixation on opponent')
    ax3.set_title('Participant is behind')
    ax3.set_xlim([135, 325])
    ax3.set_ylim([0, 1])

    ax4.plot(combined_x_data_ahead, combined_hmd_data_ahead)  # see x below
    ax4.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, color='blue', alpha=0.1, label='Fixation on road')
    ax4.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, 1, color='red', alpha=0.1,
                     label='Fixation on opponent')
    ax4.set_title('Participant is ahead')
    ax4.set_xlim([135, 325])
    ax4.set_ylim([0, 1])

    ax3.axvline(global_crt_index['median_60_40_v1'][0], 0, 1, color='r', label='Average conflict resolved')
    ax4.axvline(global_crt_index['median_60_40_v2'][0], 0, 1, color='r', label='Average conflict resolved')

    fig.text(0.05, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "Average travelled distance [m]", ha="center", va="center")
    ax3.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_behind, 2)))
    ax4.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_ahead, 2)))
    ax3.legend(loc='lower left')
    ax4.legend(loc='lower left')

    # 50-50
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    Varjo_data5 = plot_varjo(path_to_csv_50_50)

    # interpolation
    max_len = max(len(Varjo_data5[2]), len(Varjo_data5[3]))
    delta = (len(Varjo_data5[3]) - 1) / (max_len - 1)
    outp = [interpolate(Varjo_data5[3], i * delta) for i in range(max_len)]

    max_len_hmd = max(len(Varjo_data5[0]), len(Varjo_data5[1]))
    delta = (len(Varjo_data5[1]) - 1) / (max_len_hmd - 1)
    outp_hmd = [interpolate(Varjo_data5[1], i * delta) for i in range(max_len_hmd)]

    # combining data
    x_data = [Varjo_data5[2], outp]
    hmd_data = [Varjo_data5[0], outp_hmd]
    combined_x_data = list(average(x_data))
    combined_hmd_data = list(average(hmd_data))
    average_trace = sum([Varjo_data5[4], Varjo_data5[5]]) / 2

    # plotting
    fig, ax5 = plt.subplots(1, 1)
    fig.suptitle('Gaze behavior over traveled distance condition 1 (50-50 km/h)')
    ax5.plot(combined_x_data, combined_hmd_data)  # see x below
    ax5.fill_between(combined_x_data, combined_hmd_data, color='blue', alpha=0.1, label='Fixation on road')
    ax5.fill_between(combined_x_data, combined_hmd_data, 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax5.set_xlim([135, 325])
    ax5.set_ylim([0, 1])
    ax5.set(xlabel='Average traveled distance [m]', ylabel='% fixated on AOI')
    ax5.axvline(global_crt_index['median_50_50'][0], 0, 1, color='r', label='Average conflict resolved')
    ax5.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace, 2)))
    ax5.legend(loc='lower left')

    plt.show()

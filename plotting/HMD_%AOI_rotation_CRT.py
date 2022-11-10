import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
import datetime

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


def average_nested(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))

def get_timestamps(intcolumnname, data_csv):
    time = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data_csv.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time


def plot_varjo(path_to_csv_folder, condition):
    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=118,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=215,
                                               track_start_point_distance=430,
                                               track_section_length_before=304.056,
                                               track_section_length_after=200)  # goes until 400

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
        individual_travelled_distance_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_travelled_distance_vehicle2 = travelled_distance_vehicle2[list_index]
        inner_tunnel_merge_1 = []
        inner_tunnel_merge_2 = []
        for i in range(0, 1):
            index_of_tunnel_vehicle_1 = min(range(len(individual_travelled_distance_vehicle1)), key=lambda i: abs(
                individual_travelled_distance_vehicle1[i] - track.tunnel_length))
            index_of_mergepoint_vehicle_1 = min(range(len(individual_travelled_distance_vehicle1)), key=lambda i: abs(
                individual_travelled_distance_vehicle1[i] - simulation_constants.track_section_length_before))

            index_of_tunnel_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - track.tunnel_length))
            index_of_mergepoint_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - simulation_constants.track_section_length_before))

            inner_tunnel_merge_1.append(index_of_tunnel_vehicle_1)
            inner_tunnel_merge_1.append(index_of_mergepoint_vehicle_1)

            inner_tunnel_merge_2.append(index_of_tunnel_vehicle_2)
            inner_tunnel_merge_2.append(index_of_mergepoint_vehicle_2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_tunnel_merge_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_tunnel_merge_2)

    #time at merging
    time_at_merge_vehicle1 = []
    time_at_merge_vehicle2 = []
    for i in range(len(all_pds_list)):
        time_in_datetime = get_timestamps(0, all_pds_list[i])
        time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
        time_in_seconds_trail = np.array(time_in_seconds_trail)
        at_merge_vehicle1 = time_in_seconds_trail[indexes_of_tunnel_and_merge_vehicle1[i][1]]
        at_merge_vehicle2 = time_in_seconds_trail[indexes_of_tunnel_and_merge_vehicle2[i][1]]
        time_at_merge_vehicle1.append(at_merge_vehicle1)
        time_at_merge_vehicle2.append(at_merge_vehicle2)

    norm_at_merge_vehicle1 = sum(time_at_merge_vehicle1) / len(time_at_merge_vehicle1)
    norm_at_merge_vehicle2 = sum(time_at_merge_vehicle2) / len(time_at_merge_vehicle2)
    norm_at_merge_combined = (norm_at_merge_vehicle1+norm_at_merge_vehicle2) / 2


    #time at leaving tunnel.
    time_at_exit_vehicle1 = []
    time_at_exit_vehicle2 = []
    for i in range(len(all_pds_list)):
        time_in_datetime = get_timestamps(0, all_pds_list[i])
        time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
        time_in_seconds_trail = np.array(time_in_seconds_trail)
        at_exit_vehicle1 = time_in_seconds_trail[indexes_of_tunnel_and_merge_vehicle1[i][0]]
        at_exit_vehicle2 = time_in_seconds_trail[indexes_of_tunnel_and_merge_vehicle2[i][0]]
        time_at_exit_vehicle1.append(at_exit_vehicle1)
        time_at_exit_vehicle2.append(at_exit_vehicle2)

    norm_at_exit_vehicle1 = sum(time_at_exit_vehicle1) / len(time_at_exit_vehicle1)
    norm_at_exit_vehicle2 = sum(time_at_exit_vehicle2) / len(time_at_exit_vehicle2)
    norm_at_exit_combined = (norm_at_exit_vehicle1+norm_at_exit_vehicle2) / 2

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
        interactive_trace_2 = travelled_distance_vehicle1[i][
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
                if individual_hmd_rot_list_1[i] > 0.96:  # this we need to know better
                    inner_attention_list_1.append(1)
                else:
                    inner_attention_list_1.append(0)

            for i in range(len(individual_hmd_rot_list_2)):
                if individual_hmd_rot_list_2[i] > 0.96:  # this we need to know better
                    inner_attention_list_2.append(1)
                else:
                    inner_attention_list_2.append(0)

            on_ramp_vs_opponent_vehicle1.append(inner_attention_list_1)
            on_ramp_vs_opponent_vehicle2.append(inner_attention_list_2)

    average_hmd_vehicle1 = list(average_nested(on_ramp_vs_opponent_vehicle1))
    average_hmd_vehicle2 = list(average_nested(on_ramp_vs_opponent_vehicle2))

    average_hmd_combined = [(x + y)/2 for x, y in zip(average_hmd_vehicle1, average_hmd_vehicle2)]

    average_trace_vehicle = sum(average_hmd_combined) / len(average_hmd_combined)

    ysmoothed = list(gaussian_filter1d(average_hmd_combined, sigma=4))
    x = list(np.linspace(norm_at_exit_combined, norm_at_merge_combined, len(average_hmd_combined)))

    ##get median lines
    path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt.csv')
    global_crt = pd.read_csv(path_to_data_csv, sep=',')

    if condition == '50-50':
        crt = global_crt['median_50_50'][0]
        index_crt = min(range(len(x)),
                                   key=lambda i: abs(x[i] - crt))
        average_before = sum(ysmoothed[0:index_crt]) / len(ysmoothed[0:index_crt])
        average_after = sum(ysmoothed[index_crt:len(ysmoothed)]) / len(ysmoothed[index_crt:len(ysmoothed)])

    elif condition == '55-45':
        crt = global_crt['median_55_45'][0]
        index_crt = min(range(len(x)),
                                   key=lambda i: abs(x[i] - crt))
        average_before = sum(ysmoothed[0:index_crt]) / len(ysmoothed[0:index_crt])
        average_after = sum(ysmoothed[index_crt:len(ysmoothed)]) / len(ysmoothed[index_crt:len(ysmoothed)])

    elif condition == '60-40':
        crt = global_crt['median_60_40'][0]
        index_crt = min(range(len(x)),
                                   key=lambda i: abs(x[i] - crt))
        average_before = sum(ysmoothed[0:index_crt]) / len(ysmoothed[0:index_crt])
        average_after = sum(ysmoothed[index_crt:len(ysmoothed)]) / len(ysmoothed[index_crt:len(ysmoothed)])

    return x, ysmoothed, average_trace_vehicle, crt, norm_at_exit_combined, norm_at_merge_combined, average_before, average_after


if __name__ == '__main__':
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Comparison gaze behavior')

    #condition 60-40
    path_60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40'
    condition = '60-40'
    Varjo_data = plot_varjo(path_60_40, condition)

    ax1.plot(Varjo_data[0], Varjo_data[1])  # see x below
    ax1.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax1.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax1.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    ax1.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[2], 2)))
    ax1.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    ax1.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    ax1.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))


    ax1.set_title("Condition 60-40")
    ax1.set_xlim([Varjo_data[4], Varjo_data[5]])
    ax1.set_ylim([0, 1])
    # ax1.set(ylabel='% fixated on AOI')
    fig.text(0.06, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    fig.text(0.5, 0.06, "Time [s]", ha="center", va="center")
    ax1.legend(loc='lower left')


    #condition 55-45
    path_55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_55_45'
    condition = '55-45'
    Varjo_data = plot_varjo(path_55_45, condition)

    ax2.plot(Varjo_data[0], Varjo_data[1])  # see x below
    ax2.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax2.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax2.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    ax2.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[2], 2)))
    ax2.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    ax2.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    ax2.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))

    ax2.set_title("Condition 55-45")
    ax2.set_xlim(Varjo_data[4], Varjo_data[5])
    ax2.set_ylim([0, 1])
    # ax2.set(ylabel='% fixated on AOI')
    ax2.legend(loc='lower left')

    #condition 50-50
    path_50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50'
    condition = '50-50'
    Varjo_data = plot_varjo(path_50_50, condition)

    ax3.plot(Varjo_data[0], Varjo_data[1])  # see x below
    ax3.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax3.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax3.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    ax3.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[2], 2)))
    ax3.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    ax3.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    ax3.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))

    ax3.set_title("Condition 50-50")
    ax3.set_xlim(Varjo_data[4], Varjo_data[5])
    ax3.set_ylim([0, 1])
    # ax3.set(ylabel='% fixated on AOI')
    ax3.legend(loc='lower left')

    plt.show()
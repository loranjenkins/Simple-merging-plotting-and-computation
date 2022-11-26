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


def plot_varjo_combined(path_to_csv_folder, condition):
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
    fig, (ax1) = plt.subplots(1, 1)
    # fig.suptitle('Comparison gaze behavior between conditions')
    #
    # #condition 60-40
    # path_60_40 = r'D:\Thesis_data_all_experiments\Conditions\condition_60_40'
    # condition = '60-40'
    # Varjo_data = plot_varjo_combined(path_60_40, condition)
    #
    # ax1.plot(Varjo_data[0], Varjo_data[1])  # see x below
    # ax1.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    # ax1.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    #
    # ax1.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    # ax1.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[2], 2)))
    # # ax1.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    # # ax1.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    # ax1.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))
    #
    #
    # ax1.set_title("Condition 1")
    # ax1.set_xlim([Varjo_data[4], Varjo_data[5]])
    # ax1.set_ylim([0, 1])
    # # ax1.set(ylabel='% fixated on AOI')
    # fig.text(0.06, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    # fig.text(0.5, 0.06, "Time [s]", ha="center", va="center")
    # ax1.legend(loc='lower left')

    #condition 50-50
    path_50_50 = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    condition = '50-50'
    Varjo_data = plot_varjo_combined(path_50_50, condition)

    ax1.plot(Varjo_data[0], Varjo_data[1])  # see x below
    ax1.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax1.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax1.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    ax1.plot([], [], ' ', label='Average % fixation: ' + str(round(Varjo_data[2], 2)))
    # ax3.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    # ax3.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    # ax1.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))

    ax1.set_title("Gaze behavior of participants in condition 1")
    ax1.set_xlim(Varjo_data[4], Varjo_data[5])
    ax1.set_ylim([0, 1])
    ax1.set(xlabel='Time [s]', ylabel='% fixated on AOI')
    ax1.legend(loc='lower left')

    # #condition 55-45
    # path_55_45 = r'D:\Thesis_data_all_experiments\Conditions\condition_55_45'
    # condition = '55-45'
    # Varjo_data = plot_varjo_combined(path_55_45, condition)
    #
    # ax3.plot(Varjo_data[0], Varjo_data[1])  # see x below
    # ax3.fill_between(Varjo_data[0], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    # ax3.fill_between(Varjo_data[0], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    #
    # ax3.axvline(Varjo_data[3], 0, 1, color='r', label='Average CRT')
    # ax3.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[2], 2)))
    # # ax2.plot([], [], ' ', label='Average % fixated before average CRT: ' + str(round(Varjo_data[6], 2)))
    # # ax2.plot([], [], ' ', label='Average % fixated after average CRT: ' + str(round(Varjo_data[7], 2)))
    # ax3.plot([], [], ' ', label='Average CRT: ' + str(round(Varjo_data[3], 2)))
    #
    # ax3.set_title("Condition 3")
    # ax3.set_xlim(Varjo_data[4], Varjo_data[5])
    # ax3.set_ylim([0, 1])
    # # ax2.set(ylabel='% fixated on AOI')
    # ax3.legend(loc='lower left')

    plt.show()

    # path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt.csv')
    # global_crt = pd.read_csv(path_to_data_csv, sep=',')
    #
    # # left ahead 55-45
    # path_to_csv_55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\left'
    # Varjo_data = plot_varjo(path_to_csv_55_45)
    #
    # # right ahead 45-55
    # path_to_csv_45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\right'
    # Varjo_data1 = plot_varjo(path_to_csv_45_55)
    #
    # #negative headway combined -> velocity advantage
    # x_data_velocity_advantage = [Varjo_data[2], Varjo_data1[3]]
    # hmd_data_velocity_advantage = [Varjo_data[0], Varjo_data1[1]]
    # combined_x_data_advantage = list(average(x_data_velocity_advantage))
    # combined_hmd_data_advantage = list(average(hmd_data_velocity_advantage))
    # average_trace_advantage = sum([Varjo_data[4], Varjo_data1[5]]) / 2
    # time_x_start_advantage = sum([Varjo_data[6], Varjo_data1[8]]) / 2
    # time_x_end_advantage = sum([Varjo_data[7], Varjo_data1[9]]) / 2
    #
    # # positive headway combined -> velocity advantage
    # x_data_velocity_disadvantage = [Varjo_data[3], Varjo_data1[2]]
    # hmd_data_velocity_disadvantage = [Varjo_data[1], Varjo_data1[0]]
    # combined_x_data_disadvantage = list(average(x_data_velocity_disadvantage))
    # combined_hmd_data_disadvantage = list(average(hmd_data_velocity_disadvantage))
    # average_trace_disadvantage = sum([Varjo_data[5], Varjo_data1[4]]) / 2
    # time_x_start_disadvantage = min([Varjo_data[8], Varjo_data1[6]])
    # time_x_end_disadvantage = min([Varjo_data[9], Varjo_data1[7]])
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.text(0.04, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    # fig.text(0.5, 0.06, "Time [s]", ha="center", va="center")
    # fig.suptitle('Comparison gaze behavior between participants condition 3')
    # ax1.set_title('Participant has velocity advantage')
    # ax1.plot(combined_x_data_advantage, combined_hmd_data_advantage)
    # ax1.fill_between(combined_x_data_advantage, combined_hmd_data_advantage, color='blue', alpha=0.1, label='Fixation on road')
    # ax1.fill_between(combined_x_data_advantage, combined_hmd_data_advantage, 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax1.set_xlim([time_x_start_advantage, time_x_end_advantage])
    # ax1.set_ylim([0, 1])
    # ax1.plot([], [], ' ', label='Average % fixation: ' + str(round(average_trace_advantage, 2)))
    # ax1.axvline(global_crt['median_55_45'][0], 0, 1, color='r', label='Average CRT')
    # ax1.legend(loc='lower left')
    #
    #
    # ax2.set_title('Participant has velocity disadvantage')
    # ax2.plot(combined_x_data_disadvantage, combined_hmd_data_disadvantage)
    # ax2.fill_between(combined_x_data_disadvantage, combined_hmd_data_disadvantage, color='blue', alpha=0.1, label='Fixation on road')
    # ax2.fill_between(combined_x_data_disadvantage, combined_hmd_data_disadvantage, 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax2.set_xlim([time_x_start_disadvantage, 19.08])
    # ax2.set_ylim([0, 1])
    # ax2.plot([], [], ' ', label='Average % fixation: ' + str(round(average_trace_disadvantage, 2)))
    # ax2.axvline(global_crt['median_55_45'][0], 0, 1, color='r', label='Average CRT')
    # ax2.legend(loc='lower left')
    #
    #
    # # left ahead 60-40
    # path_to_csv_60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\left'
    # Varjo_data3 = plot_varjo(path_to_csv_60_40)
    #
    # # right ahead 40-60
    # path_to_csv_40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\right'
    # Varjo_data4 = plot_varjo(path_to_csv_40_60)
    #
    # #negative headway combined -> velocity advantage
    # x_data_velocity_advantage = [Varjo_data3[2], Varjo_data4[3]]
    # hmd_data_velocity_advantage = [Varjo_data3[0], Varjo_data4[1]]
    # combined_x_data_advantage = list(average(x_data_velocity_advantage))
    # combined_hmd_data_advantage = list(average(hmd_data_velocity_advantage))
    # average_trace_advantage = sum([Varjo_data3[4], Varjo_data4[5]]) / 2
    # time_x_start_advantage = min([Varjo_data3[6], Varjo_data4[8]])
    # time_x_end_advantage = min([Varjo_data3[7], Varjo_data4[9]])
    #
    # # positive headway combined -> velocity advantage
    # x_data_velocity_disadvantage = [Varjo_data3[3], Varjo_data4[2]]
    # hmd_data_velocity_disadvantage = [Varjo_data3[1], Varjo_data4[0]]
    # combined_x_data_disadvantage = list(average(x_data_velocity_disadvantage))
    # combined_hmd_data_disadvantage = list(average(hmd_data_velocity_disadvantage))
    # average_trace_disadvantage = sum([Varjo_data3[5], Varjo_data4[4]]) / 2
    # time_x_start_disadvantage = min([Varjo_data3[8], Varjo_data4[6]])
    # time_x_end_disadvantage = min([Varjo_data3[9], Varjo_data4[7]])
    #
    # fig, (ax3, ax4) = plt.subplots(1, 2)
    # fig.text(0.04, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    # fig.text(0.5, 0.06, "Time [s]", ha="center", va="center")
    # fig.suptitle('Comparison gaze behavior between participants condition 1')
    # ax3.set_title('Participant has velocity advantage')
    # ax3.plot(combined_x_data_advantage, combined_hmd_data_advantage)
    # ax3.fill_between(combined_x_data_advantage, combined_hmd_data_advantage, color='blue', alpha=0.1, label='Fixation on road')
    # ax3.fill_between(combined_x_data_advantage, combined_hmd_data_advantage, 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax3.set_xlim([time_x_start_advantage, time_x_end_advantage])
    # ax3.set_ylim([0, 1])
    # ax3.plot([], [], ' ', label='Average % fixation: ' + str(round(average_trace_advantage, 2)))
    # ax3.axvline(global_crt['median_60_40'][0], 0, 1, color='r', label='Average CRT')
    # ax3.legend(loc='lower left')
    #
    #
    #
    # ax4.set_title('Participant has velocity disadvantage')
    # ax4.plot(combined_x_data_disadvantage, combined_hmd_data_disadvantage)
    # ax4.fill_between(combined_x_data_disadvantage, combined_hmd_data_disadvantage, color='blue', alpha=0.1, label='Fixation on road')
    # ax4.fill_between(combined_x_data_disadvantage, combined_hmd_data_disadvantage, 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax4.set_xlim([time_x_start_disadvantage, time_x_end_disadvantage])
    # ax4.set_ylim([0, 1])
    # ax4.plot([], [], ' ', label='Average % fixation: ' + str(round(average_trace_disadvantage, 2)))
    # ax4.axvline(global_crt['median_60_40'][0], 0, 1, color='r', label='Average CRT')
    # ax4.legend(loc='lower left')
    #
    #
    # # 50-50
    # path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    # Varjo_data = plot_varjo(path_to_csv_50_50)
    #
    # fig, (ax5, ax6) = plt.subplots(1, 2)
    # fig.text(0.04, 0.5, "% fixated on AOI", va='center', rotation='vertical')
    # fig.text(0.5, 0.06, "Time [s]", ha="center", va="center")
    # fig.suptitle('Comparison gaze behavior between participants condition 2')
    # ax5.plot(Varjo_data[2], Varjo_data[0])  # see x below
    # ax5.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    # ax5.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax5.set_title('Participant has equal velocity')
    # ax5.set_xlim([Varjo_data[6], Varjo_data[7]])
    # ax5.set_ylim([0, 1])
    #
    # ax6.plot(Varjo_data[3], Varjo_data[1])  # see x below
    # ax6.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    # ax6.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax6.set_title('Participant has equal velocity')
    # ax6.set_xlim([Varjo_data[8], Varjo_data[9]])
    # ax6.set_ylim([0, 1])
    #
    # ax5.axvline(global_crt['median_50_50'][0], 0, 1, color='r', label='Average CRT')
    # ax6.axvline(global_crt['median_50_50'][0], 0, 1, color='r', label='Average CRT')
    #
    # ax5.plot([], [], ' ', label='Average % fixation: ' + str(round(Varjo_data[4], 2)))
    # ax6.plot([], [], ' ', label='Average % fixation: ' + str(round(Varjo_data[5], 2)))
    # ax5.legend(loc='lower left')
    # ax6.legend(loc='lower left')
    #
    # plt.show()

    # --- old with interpolation
    # # negative headway combined -> velocity advantage
    # x_data_velocity_advantage = [Varjo_data[2], Varjo_data1[3]]
    # hmd_data_velocity_advantage = [Varjo_data[0], Varjo_data1[1]]
    # combined_x_data_advantage = list(average(x_data_velocity_advantage))
    # combined_hmd_data_advantage = list(average(hmd_data_velocity_advantage))
    # average_trace_advantage = sum([Varjo_data[4], Varjo_data1[5]]) / 2
    # time_x_start_advantage = sum([Varjo_data[6], Varjo_data1[8]]) / 2
    # time_x_end_advantage = sum([Varjo_data[7], Varjo_data1[9]]) / 2
    # # positive headway combined -> velocity advantage
    # x_data_velocity_disadvantage = [Varjo_data[3], Varjo_data1[2]]
    # hmd_data_velocity_disadvantage = [Varjo_data[1], Varjo_data1[0]]
    # combined_x_data_disadvantage = list(average(x_data_velocity_disadvantage))
    # combined_hmd_data_disadvantage = list(average(hmd_data_velocity_disadvantage))
    # average_trace_disadvantage = sum([Varjo_data[5], Varjo_data1[4]]) / 2
    # time_x_start_disadvantage = min([Varjo_data[8], Varjo_data1[6]])
    # time_x_end_disadvantage = min([Varjo_data[9], Varjo_data1[7]])

    # 60-40
    # # negative headway combined -> velocity advantage
    # x_data_velocity_advantage = [Varjo_data3[2], Varjo_data4[3]]
    # hmd_data_velocity_advantage = [Varjo_data3[0], Varjo_data4[1]]
    # combined_x_data_advantage = list(average(x_data_velocity_advantage))
    # combined_hmd_data_advantage = list(average(hmd_data_velocity_advantage))
    # average_trace_advantage = sum([Varjo_data3[4], Varjo_data4[5]]) / 2
    # time_x_start_advantage = min([Varjo_data3[6], Varjo_data4[8]])
    # time_x_end_advantage = min([Varjo_data3[7], Varjo_data4[9]])

    # # positive headway combined -> velocity advantage
    # x_data_velocity_disadvantage = [Varjo_data3[3], Varjo_data4[2]]
    # hmd_data_velocity_disadvantage = [Varjo_data3[1], Varjo_data4[0]]
    # combined_x_data_disadvantage = list(average(x_data_velocity_disadvantage))
    # combined_hmd_data_disadvantage = list(average(hmd_data_velocity_disadvantage))
    # average_trace_disadvantage = sum([Varjo_data3[5], Varjo_data4[4]]) / 2
    # time_x_start_disadvantage = min([Varjo_data3[8], Varjo_data4[6]])
    # time_x_end_disadvantage = min([Varjo_data3[9], Varjo_data4[7]])

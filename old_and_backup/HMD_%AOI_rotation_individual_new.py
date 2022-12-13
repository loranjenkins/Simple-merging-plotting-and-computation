import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from scipy import interpolate
import datetime
import os
import numpy as np
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

def get_timestamps(intcolumnname, data_csv):
    time = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data_csv.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time

def plot_varjo(path_to_csv_folder):
    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=125,
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)


    dict = {'time_vehicle1': [],
            'time_vehicle2': [],
            'traveled_distance_vehicle1': [],
            'traveled_distance_vehicle2': [],
            'gaze_vehicle1': [],
            'gaze_vehicle2': [],
            'trail': []}

    files_directory = path_to_csv_folder
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

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
                individual_travelled_distance_vehicle1[i] - track.section_length_before))

            index_of_tunnel_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - track.tunnel_length))
            index_of_mergepoint_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - track.section_length_before))

            inner_tunnel_merge_1.append(index_of_tunnel_vehicle_1)
            inner_tunnel_merge_1.append(index_of_mergepoint_vehicle_1)

            inner_tunnel_merge_2.append(index_of_tunnel_vehicle_2)
            inner_tunnel_merge_2.append(index_of_mergepoint_vehicle_2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_tunnel_merge_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_tunnel_merge_2)

    path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt.csv')
    global_crt_median = pd.read_csv(path_to_data_csv, sep=',')


    # compute travelled distance for indexes tunnel and merge
    interactive_trace_vehicle1 = []
    interactive_trace_vehicle2 = []
    for list_index in range(len(travelled_distance_vehicle1)):
        individual_travelled_distance_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_travelled_distance_vehicle2 = travelled_distance_vehicle2[list_index]
        individual_travelled_distance_vehicle1 = individual_travelled_distance_vehicle1[
                      indexes_of_tunnel_and_merge_vehicle1[list_index][0]:indexes_of_tunnel_and_merge_vehicle1[list_index][1]]
        individual_travelled_distance_vehicle2 = individual_travelled_distance_vehicle2[
                      indexes_of_tunnel_and_merge_vehicle2[list_index][0]:indexes_of_tunnel_and_merge_vehicle2[list_index][1]]
        interactive_trace_vehicle1.append(individual_travelled_distance_vehicle1)
        interactive_trace_vehicle2.append(individual_travelled_distance_vehicle2)

    for i in range(len(interactive_trace_vehicle1)):
        inner = interactive_trace_vehicle1[i]
        for value in inner:
            dict['traveled_distance_vehicle1'].append(value)

    for i in range(len(interactive_trace_vehicle2)):
        inner = interactive_trace_vehicle2[i]
        for value in inner:
            dict['traveled_distance_vehicle2'].append(value)


    time_in_seconds_trails_ = []
    for i in range(len(all_pds_list)):
        time_in_datetime = get_timestamps(0, all_pds_list[i])
        time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
        time_in_seconds_trail = time_in_seconds_trail
        time_in_seconds_trails_.append(time_in_seconds_trail)

    # total time in interactive area for each vehicle
    time_in_seconds_trails_v1 = []
    time_in_seconds_trails_v2 = []
    per_trail = []
    for i in range(len(time_in_seconds_trails_)):
        inner = time_in_seconds_trails_[i]
        new_list_v1 = inner[
                      indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]]
        new_list_v2 = inner[
                      indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]]
        together = list([new_list_v1, new_list_v2])
        per_trail.append(together)
        time_in_seconds_trails_v1.append(new_list_v1)
        time_in_seconds_trails_v2.append(new_list_v2)

    for i in range(len(per_trail)):
        max_len = max(len(item) for item in per_trail[i])
        for item in per_trail[i]:
            if len(item) < max_len:
                item.extend([np.nan] * (max_len - len(item)))

    trial = 1
    for i in range(len(time_in_seconds_trails_v1)):
        # inner = len(time_in_seconds_trails_v1[i])
        # print(inner)
        dict['trail'] += [trial] * len(time_in_seconds_trails_v1[i])
        trial += 1

    for i in range(len(time_in_seconds_trails_v1)):
        inner = time_in_seconds_trails_v1[i]
        for value in inner:
            dict['time_vehicle1'].append(value)

    for i in range(len(time_in_seconds_trails_v2)):
        inner = time_in_seconds_trails_v2[i]
        for value in inner:
            dict['time_vehicle2'].append(value)

    # # interactive gaze data for each vehicle
    hmd_rot_interactive_area_vehicle1 = []
    hmd_rot_interactive_area_vehicle2 = []

    for i in range(len(indexes_of_tunnel_and_merge_vehicle1)):
        hmd_rot_1 = list(all_pds_list[i]['HMD_rotation_vehicle1'][
                         indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]])

        hmd_rot_2 = list(all_pds_list[i]['HMD_rotation_vehicle2'][
                         indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]])
        hmd_rot_interactive_area_vehicle1.append(hmd_rot_1)
        hmd_rot_interactive_area_vehicle2.append(hmd_rot_2)

    # change hmd rots to ones and zeros
    on_ramp_vs_opponent_vehicle1 = []
    on_ramp_vs_opponent_vehicle2 = []
    for list_index in range(len(hmd_rot_interactive_area_vehicle1)):
        individual_hmd_rot_list_1 = hmd_rot_interactive_area_vehicle1[list_index]
        individual_hmd_rot_list_2 = hmd_rot_interactive_area_vehicle2[list_index]

        inner_attention_list_1 = []
        inner_attention_list_2 = []

        for hmd_rot in range(len(individual_hmd_rot_list_1)):
            if individual_hmd_rot_list_1[hmd_rot] > 0.95:  # this we need to know better
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

    for i in range(len(on_ramp_vs_opponent_vehicle1)):
        inner = on_ramp_vs_opponent_vehicle1[i]
        for value in inner:
            dict['gaze_vehicle1'].append(value)

    for i in range(len(on_ramp_vs_opponent_vehicle2)):
        inner = on_ramp_vs_opponent_vehicle2[i]
        for value in inner:
            dict['gaze_vehicle2'].append(value)

    return dict

if __name__ == '__main__':
    # 50-50
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    dict_50_50 = plot_varjo(path_to_csv_50_50)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_50_50.items()]))
    df['Average_traveled'] = df[["time_vehicle1", "time_vehicle2"]].mean(axis=1)
    df['Average_fixation'] = df[["gaze_vehicle1", "gaze_vehicle2"]].mean(axis=1)
    min_traveled_equal = min(min(df['time_vehicle1']), min(df['time_vehicle2']))
    max_traveled_equal = max(max(df['time_vehicle1']), max(df['time_vehicle2']))

    new_traveled = np.linspace(min_traveled_equal, max_traveled_equal, 2000)

    new_df = {'trial': [],
              'new_traveled': [],
              'new_data': []}

    for trial_number in df['trail'].unique():
        trial_data = df.loc[df['trail'] == trial_number, :]

        new_df['new_traveled'] += list(new_traveled)
        new_df['new_data'] += list(np.interp(new_traveled, trial_data['Average_traveled'],
                                             trial_data['Average_fixation']))
        new_df['trial'] += [trial_number] * len(new_traveled)

    new_df = pd.DataFrame(new_df)

    on_road_fixation_before = list(new_df['new_data']).count(1)
    both_else_fixation_before = list(new_df['new_data']).count(0.5)
    on_opponent_fixation_before = list(new_df['new_data']).count(0)

    average_fixation = (on_opponent_fixation_before + both_else_fixation_before / 2) / sum(
        [on_road_fixation_before, both_else_fixation_before, on_opponent_fixation_before])

    fig, ax5 = plt.subplots(1, 1)
    fig.suptitle('Gaze behavior before-after the CRT condition 1 (50-50 km/h)')

    line_combined = sns.lineplot(x="new_traveled", y="new_data", data=new_df)

    l1 = line_combined.lines[0]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]

    ax5.fill_between(x1, y1, color='blue', alpha=0.1, label='Fixation on road')
    ax5.fill_between(x1, y1, 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax5.set_xlim([min_traveled_equal, max_traveled_equal])
    ax5.set_ylim([0, 1])

    ax5.plot([], [], ' ', label='Average fixation: ' + str(round(average_fixation, 2)))
    ax5.set(xlabel='CRT [s]', ylabel='Fixation on opponent [%]')
    ax5.legend(loc='lower left')

    plt.show()



# # interpolation
    # max_len = max(len(Varjo_data5[2]), len(Varjo_data5[3]))
    # delta = (len(Varjo_data5[3]) - 1) / (max_len - 1)
    # outp = [interpolate(Varjo_data5[3], i * delta) for i in range(max_len)]
    #
    # max_len_hmd = max(len(Varjo_data5[0]), len(Varjo_data5[1]))
    # delta = (len(Varjo_data5[1]) - 1) / (max_len_hmd - 1)
    # outp_hmd = [interpolate(Varjo_data5[1], i * delta) for i in range(max_len_hmd)]
    #
    # # combining data
    # x_data = [Varjo_data5[2], outp]
    # hmd_data = [Varjo_data5[0], outp_hmd]
    # combined_x_data = list(average(x_data))
    # combined_hmd_data = list(average(hmd_data))
    # average_trace = sum([Varjo_data5[4], Varjo_data5[5]]) / 2
    #
    # # plotting
    # fig, ax5 = plt.subplots(1, 1)
    # fig.suptitle('Gaze behavior over traveled distance condition 1 (50-50 km/h)')
    # ax5.plot(combined_x_data, combined_hmd_data)  # see x below
    # ax5.fill_between(combined_x_data, combined_hmd_data, color='blue', alpha=0.1, label='Fixation on road')
    # ax5.fill_between(combined_x_data, combined_hmd_data, 1, color='red', alpha=0.1, label='Fixation on opponent')
    # ax5.set_xlim([125, 325])
    # ax5.set_ylim([0, 1])
    # ax5.set(xlabel='Traveled distance [m]', ylabel='Fixation on AOI [%]')
    # # ax5.axvline(global_crt_index['median_50_50'][0], 0, 1, color='r', label='Average conflict resolved')
    # ax5.axvline(global_crt_index['median_50_50'][0], 0, 1, color='r', label='Kernel density maximum')
    #
    # ax5.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace, 2)))
    # ax5.legend(loc='lower left')
    #
    # plt.show()




    # path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt_index.csv')
    # global_crt_index = pd.read_csv(path_to_data_csv, sep=',')
    #
    # # --------------------------------------------------
    # # #left ahead 45-55
    # path_to_csv_55_45 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    # Varjo_data = plot_varjo(path_to_csv_55_45)
    # # right ahead 45-55
    # path_to_csv_45_55 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    # Varjo_data1 = plot_varjo(path_to_csv_45_55)
    #
    # #interpolation
    # max_len_x_behind = max(len(Varjo_data[2]), len(Varjo_data1[3]))
    # delta = (len(Varjo_data[2]) - 1) / (max_len_x_behind - 1)
    # outp_x_behind = [interpolate(Varjo_data[2], i * delta) for i in range(max_len_x_behind)]
    #
    # max_len_hmd_behind = max(len(Varjo_data[0]), len(Varjo_data1[1]))
    # delta = (len(Varjo_data[0]) - 1) / (max_len_hmd_behind - 1)
    # outp_hmd_behind = [interpolate(Varjo_data[0], i * delta) for i in range(max_len_hmd_behind)]
    #
    # x_data_behind = [outp_x_behind, Varjo_data1[3]]
    # hmd_data_behind = [outp_hmd_behind, Varjo_data1[1]]
    # combined_x_data_behind = list(average(x_data_behind))
    # combined_hmd_data_behind = list(average(hmd_data_behind))
    # average_trace_behind = sum([Varjo_data[4], Varjo_data1[5]]) / 2
    #
    # #interpolation
    # max_len_x_ahead = max(len(Varjo_data[3]), len(Varjo_data1[2]))
    # delta = (len(Varjo_data[3]) - 1) / (max_len_x_ahead - 1)
    # outp_x_ahead = [interpolate(Varjo_data[3], i * delta) for i in range(max_len_x_ahead)]
    #
    # max_len_hmd_ahead = max(len(Varjo_data[1]), len(Varjo_data1[0]))
    # delta = (len(Varjo_data[1]) - 1) / (max_len_hmd_ahead - 1)
    # outp_hmd_ahead = [interpolate(Varjo_data[1], i * delta) for i in range(max_len_hmd_ahead)]
    #
    # x_data_ahead = [outp_x_ahead, Varjo_data1[2]]
    # hmd_data_velocity_ahead = [outp_hmd_ahead, Varjo_data1[0]]
    # combined_x_data_ahead = list(average(x_data_ahead))
    # combined_hmd_data_ahead = list(average(hmd_data_velocity_ahead))
    # average_trace_ahead = sum([Varjo_data[5], Varjo_data1[4]]) / 2
    #
    # #plotting
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # fig.suptitle('Comparison gaze behavior over traveled distance condition 2 (55-45 km/h)')
    # ax1.plot(combined_x_data_behind, combined_hmd_data_behind)  # see x below
    # ax1.fill_between(combined_x_data_behind, combined_hmd_data_behind, color='blue', alpha=0.1,
    #                  label='Fixation on road')
    # ax1.fill_between(combined_x_data_behind, combined_hmd_data_behind, 1, color='red', alpha=0.1,
    #                  label='Fixation on opponent')
    # ax1.set_title('Participant is behind')
    # ax1.set_xlim([125, 325])
    # ax1.set_ylim([0, 1])
    #
    # ax2.plot(combined_x_data_ahead, combined_hmd_data_ahead)  # see x below
    # ax2.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, color='blue', alpha=0.1, label='Fixation on road')
    # ax2.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, 1, color='red', alpha=0.1,
    #                  label='Fixation on opponent')
    # ax2.set_title('Participant is ahead')
    # ax2.set_xlim([125, 325])
    # ax2.set_ylim([0, 1])
    #
    # # ax1.axvline(global_crt_index['median_55_45_v1'][0], 0, 1, color='r', label='Average conflict resolved')
    # # ax2.axvline(global_crt_index['median_55_45_v2'][0], 0, 1, color='r', label='Average conflict resolved')
    # ax1.axvline(global_crt_index['median_55_45_v1'][0], 0, 1, color='r', label='Kernel density maximum')
    # ax2.axvline(global_crt_index['median_55_45_v2'][0], 0, 1, color='r', label='Kernel density maximum')
    #
    # fig.text(0.05, 0.5, "Fixation on AOI [%]", va='center', rotation='vertical')
    # fig.text(0.5, 0.03, "Traveled distance [m]", ha="center", va="center")
    # ax1.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_behind, 2)))
    # ax2.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_ahead, 2)))
    # ax1.legend(loc='lower left')
    # ax2.legend(loc='lower left')
    #
    # # left ahead 60-40
    # path_to_csv_40_60 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'
    # Varjo_data3 = plot_varjo(path_to_csv_40_60)
    # path_to_csv_60_40 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'
    # Varjo_data4 = plot_varjo(path_to_csv_60_40)
    #
    # #interpolation
    # max_len_x_behind = max(len(Varjo_data3[2]), len(Varjo_data4[3]))
    # delta = (len(Varjo_data3[2]) - 1) / (max_len_x_behind - 1)
    # outp_x_behind = [interpolate(Varjo_data3[2], i * delta) for i in range(max_len_x_behind)]
    #
    # max_len_hmd_behind = max(len(Varjo_data3[0]), len(Varjo_data4[1]))
    # delta = (len(Varjo_data3[0]) - 1) / (max_len_hmd_behind - 1)
    # outp_hmd_behind = [interpolate(Varjo_data3[0], i * delta) for i in range(max_len_hmd_behind)]
    #
    # x_data_behind = [outp_x_behind, Varjo_data4[3]]
    # hmd_data_behind = [outp_hmd_behind, Varjo_data4[1]]
    # combined_x_data_behind = list(average(x_data_behind))
    # combined_hmd_data_behind = list(average(hmd_data_behind))
    # average_trace_behind = sum([Varjo_data3[4], Varjo_data4[5]]) / 2
    #
    # #interpolation
    # max_len_x_ahead = max(len(Varjo_data3[3]), len(Varjo_data4[2]))
    # delta = (len(Varjo_data4[3]) - 1) / (max_len_x_ahead - 1)
    # outp_x_ahead = [interpolate(Varjo_data4[3], i * delta) for i in range(max_len_x_ahead)]
    #
    # max_len_hmd_ahead = max(len(Varjo_data3[1]), len(Varjo_data4[0]))
    # delta = (len(Varjo_data4[1]) - 1) / (max_len_hmd_ahead - 1)
    # outp_hmd_ahead = [interpolate(Varjo_data4[1], i * delta) for i in range(max_len_hmd_ahead)]
    #
    # x_data_ahead = [Varjo_data3[3], outp_x_ahead]
    # hmd_data_velocity_ahead = [Varjo_data3[1], outp_hmd_ahead]
    # combined_x_data_ahead = list(average(x_data_ahead))
    # combined_hmd_data_ahead = list(average(hmd_data_velocity_ahead))
    # average_trace_ahead = sum([Varjo_data3[5], Varjo_data4[4]]) / 2
    #
    # #plotting
    # fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
    # fig.suptitle('Comparison gaze behavior over traveled distance condition 3 (60-40 km/h)')
    # ax3.plot(combined_x_data_behind, combined_hmd_data_behind)  # see x below
    # ax3.fill_between(combined_x_data_behind, combined_hmd_data_behind, color='blue', alpha=0.1,
    #                  label='Fixation on road')
    # ax3.fill_between(combined_x_data_behind, combined_hmd_data_behind, 1, color='red', alpha=0.1,
    #                  label='Fixation on opponent')
    # ax3.set_title('Participant is behind')
    # ax3.set_xlim([125, 325])
    # ax3.set_ylim([0, 1])
    #
    # ax4.plot(combined_x_data_ahead, combined_hmd_data_ahead)  # see x below
    # ax4.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, color='blue', alpha=0.1, label='Fixation on road')
    # ax4.fill_between(combined_x_data_ahead, combined_hmd_data_ahead, 1, color='red', alpha=0.1,
    #                  label='Fixation on opponent')
    # ax4.set_title('Participant is ahead')
    # ax4.set_xlim([125, 325])
    # ax4.set_ylim([0, 1])
    #
    # # ax3.axvline(global_crt_index['median_60_40_v1'][0], 0, 1, color='r', label='Average conflict resolved')
    # # ax4.axvline(global_crt_index['median_60_40_v2'][0], 0, 1, color='r', label='Average conflict resolved')
    # ax3.axvline(global_crt_index['median_60_40_v1'][0], 0, 1, color='r', label='Kernel density maximum')
    # ax4.axvline(global_crt_index['median_60_40_v2'][0], 0, 1, color='r', label='Kernel density maximum')
    #
    # fig.text(0.05, 0.5, "Fixation on AOI [%]", va='center', rotation='vertical')
    # fig.text(0.5, 0.03, "Travelled distance [m]", ha="center", va="center")
    # ax3.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_behind, 2)))
    # ax4.plot([], [], ' ', label='Average % fixation: ' + str(round(1 - average_trace_ahead, 2)))
    # ax3.legend(loc='lower left')
    # ax4.legend(loc='lower left')



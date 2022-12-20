import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import os
import numpy as np
import datetime
from natsort import natsorted
import seaborn as sns
import scipy.stats as stats

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


def get_timestamps(intcolumnname, data_csv):
    time = []
    for i in range(len(data_csv.iloc[:, intcolumnname])):
        epoch_in_nanoseconds = data_csv.iloc[i, intcolumnname]
        epoch_in_seconds = epoch_in_nanoseconds / 1000000000
        datetimes = datetime.datetime.fromtimestamp(epoch_in_seconds)
        time.append(datetimes)
    return time


def plot_varjo(path_to_csv_folder, condition, who_ahead):
    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=125,
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)

    dict = {'time_vehicle1': [], 'time_vehicle2': [], 'gaze_vehicle1': [], 'gaze_vehicle2': [], 'trial': [], 'CRT': []}

    files_directory = path_to_csv_folder
    trails = []
    for file in Path(files_directory).glob('*.csv'):
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

    time_in_seconds_trails_ = []
    for i in range(len(all_pds_list)):
        time_in_datetime = get_timestamps(0, all_pds_list[i])
        time_in_seconds_trail = [(a - time_in_datetime[0]).total_seconds() for a in time_in_datetime]
        time_in_seconds_trail = time_in_seconds_trail
        time_in_seconds_trails_.append(time_in_seconds_trail)

    # total time in interactive area for each vehicle
    time_in_seconds_trails_v1 = []
    time_in_seconds_trails_v2 = []
    for i in range(len(time_in_seconds_trails_)):
        inner = time_in_seconds_trails_[i]
        new_list_v1 = inner[
                      indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]]
        new_list_v2 = inner[
                      indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]]

        together = [new_list_v1, new_list_v2]

        max_len = max([len(item) for item in together])

        if len(together[0]) > len(together[1]):
            together[1].extend([np.nan] * (max_len - len(together[1])))
            time_in_seconds_trails_v2.append(together[1])
            time_in_seconds_trails_v1.append(together[0])

        if len(together[0]) < len(together[1]):
            together[0].extend([np.nan] * (max_len - len(together[0])))
            time_in_seconds_trails_v1.append(together[0])
            time_in_seconds_trails_v2.append(together[1])

    for i in range(len(time_in_seconds_trails_v1)):
        inner = time_in_seconds_trails_v1[i]
        for value in inner:
            dict['time_vehicle1'].append(value)

    for i in range(len(time_in_seconds_trails_v2)):
        inner = time_in_seconds_trails_v2[i]
        for value in inner:
            dict['time_vehicle2'].append(value)

    path_to_data_csv = os.path.join('..', 'data_folder', 'crt_who_is_ahead.csv')
    global_crt = pd.read_csv(path_to_data_csv, sep=',')

    trial = 1
    if condition == '50-50':
        for i in range(len(time_in_seconds_trails_v1)):
            dict['trial'] += [trial] * len(time_in_seconds_trails_v1[i])
            trial += 1

    trial_55_45_first = 1
    trial_55_45_second = 54
    if condition == '55-45':
        if who_ahead == 'vehicle1':
            for i in range(len(time_in_seconds_trails_v1)):
                dict['trial'] += [trial_55_45_first] * len(time_in_seconds_trails_v1[i])
                trial_55_45_first += 1
        elif who_ahead == 'vehicle2':
            for i in range(len(time_in_seconds_trails_v1)):
                dict['trial'] += [trial_55_45_second] * len(time_in_seconds_trails_v1[i])
                trial_55_45_second += 1

    trial_60_40_first = 1
    trial_60_40_second = 53
    if condition == '60-40':
        if who_ahead == 'vehicle1':
            for i in range(len(time_in_seconds_trails_v1)):
                dict['trial'] += [trial_60_40_first] * len(time_in_seconds_trails_v1[i])
                trial_60_40_first += 1
        elif who_ahead == 'vehicle2':
            for i in range(len(time_in_seconds_trails_v1)):
                dict['trial'] += [trial_60_40_second] * len(time_in_seconds_trails_v1[i])
                trial_60_40_second += 1

    if condition == '50-50':
        if who_ahead == 'equal':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt['crt_50_50'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

    if condition == '55-45':
        if who_ahead == 'vehicle1':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt['crt_45_55_vehicle1'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1
        elif who_ahead == 'vehicle2':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt['crt_55_45_vehicle2'][0 + index]
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

    if condition == '60-40':
        if who_ahead == 'vehicle1':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt['crt_40_60_vehicle1'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1
        elif who_ahead == 'vehicle2':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt['crt_60_40_vehicle2'][0 + index]
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

    # # interactive gaze data for each vehicle
    hmd_rot_interactive_area_vehicle1 = []
    hmd_rot_interactive_area_vehicle2 = []

    for i in range(len(all_pds_list)):
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

        together = [inner_attention_list_1, inner_attention_list_2]

        max_len = max([len(item) for item in together])

        if len(together[0]) > len(together[1]):
            together[1].extend([np.nan] * (max_len - len(together[1])))
            on_ramp_vs_opponent_vehicle2.append(together[1])
            on_ramp_vs_opponent_vehicle1.append(together[0])

        if len(together[0]) < len(together[1]):
            together[0].extend([np.nan] * (max_len - len(together[0])))
            on_ramp_vs_opponent_vehicle1.append(together[0])
            on_ramp_vs_opponent_vehicle2.append(together[1])

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
    pd.set_option('display.max_columns', None)

    # # ## -----------

    # # 55_45
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    dict55_45_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '55-45', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    dict55_45_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '55-45', 'vehicle2')

    # path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45 - kopie\vehicle 1'
    # dict55_45_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '55-45', 'vehicle1')
    #
    # path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45 - kopie\vehicle 2'
    # dict55_45_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '55-45', 'vehicle2')

    # df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v1_ahead.items()]))
    df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v2_ahead.items()]))
    # print(df1)
    # for ahead
    ahead_fixations_55_45 = pd.concat([df1['gaze_vehicle1'], df2['gaze_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_fixations')
    ahead_time_55_45 = pd.concat([df1['time_vehicle1'], df2['time_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_time')
    ahead_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0, ignore_index=True)
    ahead_trails = pd.concat([df1['trial'], df2['trial']], axis=0, ignore_index=True)
    df_ahead_55_45 = pd.concat([ahead_fixations_55_45, ahead_time_55_45, ahead_crt_55_45, ahead_trails],
                               axis=1)
    df_ahead_55_45['Average_time_minCRT'] = df_ahead_55_45['ahead_time'] - df_ahead_55_45['CRT']

    dict = {'fixations_before': [], 'fixations_after': [], 'time_before': [], 'time_after': []}
    for trial_number in df_ahead_55_45['trial'].unique():
        trial_data = df_ahead_55_45.loc[df_ahead_55_45['trial'] == trial_number]
        time_idx_before = trial_data['Average_time_minCRT'].sub(-5).abs().idxmin()
        time_idx_after = trial_data['Average_time_minCRT'].sub(5).abs().idxmin()
        average_time_dot_before = df_ahead_55_45['ahead_time'].iloc[time_idx_before]
        average_time_dot_after = df_ahead_55_45['ahead_time'].iloc[time_idx_after]

        dict['time_before'].append(average_time_dot_before)
        dict['time_after'].append(average_time_dot_after)

        fixations_before = trial_data[trial_data['Average_time_minCRT'].between(-5, 0)]
        # print(fixations_before)
        if fixations_before.empty:
            average_fixation_before = np.nan
        else:
            average_fixation_before = 1 - (
                    sum(fixations_before['ahead_fixations']) / len(fixations_before['ahead_fixations']))

        fixations_after = trial_data[trial_data['Average_time_minCRT'].between(0, 5)]
        average_fixation_after = 1 - (
                sum(fixations_after['ahead_fixations']) / len(fixations_after['ahead_fixations']))

        dict['fixations_before'].append(average_fixation_before)
        dict['fixations_after'].append(average_fixation_after)

    df_before_after_ahead_55_45 = pd.DataFrame(dict)
    df_before_ahead_55_45 = df_before_after_ahead_55_45[['fixations_before', 'time_before']].dropna()
    df_after_ahead_55_45 = df_before_after_ahead_55_45[['fixations_after', 'time_after']].dropna()

    r, p = stats.pearsonr(df_before_ahead_55_45['time_before'], df_before_ahead_55_45['fixations_before'])
    r1, p1 = stats.pearsonr(df_after_ahead_55_45['time_after'], df_after_ahead_55_45['fixations_after'])

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # fig.suptitle('Linear regression analysis before-after the CRT participant ahead condition 2 (55-45 km/h)')
    # fig.text(0.05, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    # fig.text(0.5, 0.05, "Time [s]", ha="center", va="center")
    #
    # sns.regplot(df_before_ahead_55_45, x='time_before', y='fixations_before', ax=axes[0])
    # sns.regplot(df_after_ahead_55_45, x='time_after', y='fixations_after', ax=axes[1])
    #
    # axes[0].set(xlabel=None, ylabel=None)
    # axes[1].set(xlabel=None, ylabel=None)
    #
    # axes[0].set_title('5 seconds before the CRT')
    # axes[1].set_title('5 seconds after the CRT')
    #
    # axes[0].plot([], [], ' ', label='r: ' + str(round(r, 2)))
    # axes[0].plot([], [], ' ', label='p: ' + "{:.2e}".format(p))
    # axes[1].plot([], [], ' ', label='r: ' + str(round(r1, 2)))
    # axes[1].plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))
    #
    # axes[0].legend(loc='upper left')
    # axes[1].legend(loc='upper left')


    # --------------------------------------------------------------
    # for behind
    behind_fixations_55_45 = pd.concat([df1['gaze_vehicle2'], df2['gaze_vehicle1']], axis=0, ignore_index=True).rename(
        'behind_fixations')
    behind_time_55_45 = pd.concat([df1['time_vehicle2'], df2['time_vehicle1']], axis=0, ignore_index=True).rename(
        'behind_time')
    behind_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0, ignore_index=True)
    behind_trails = pd.concat([df1['trial'], df2['trial']], axis=0, ignore_index=True)
    df_behind_55_45 = pd.concat([behind_fixations_55_45, behind_time_55_45, behind_crt_55_45, behind_trails],
                                axis=1)
    df_behind_55_45['Average_time_minCRT'] = df_behind_55_45['behind_time'] - df_behind_55_45['CRT']

    dict = {'fixations_before': [], 'fixations_after': [], 'time_before': [], 'time_after': []}
    for trial_number in df_behind_55_45['trial'].unique():
        trial_data = df_behind_55_45.loc[df_behind_55_45['trial'] == trial_number]
        time_idx_before = trial_data['Average_time_minCRT'].sub(-5).abs().idxmin()
        time_idx_after = trial_data['Average_time_minCRT'].sub(5).abs().idxmin()
        average_time_dot_before = df_behind_55_45['behind_time'].iloc[time_idx_before]
        average_time_dot_after = df_behind_55_45['behind_time'].iloc[time_idx_after]

        dict['time_before'].append(average_time_dot_before)
        dict['time_after'].append(average_time_dot_after)

        fixations_before = trial_data[trial_data['Average_time_minCRT'].between(-5, 0)]
        # print(fixations_before)
        if fixations_before.empty:
            average_fixation_before = np.nan
        else:
            average_fixation_before = 1 - (
                    sum(fixations_before['behind_fixations']) / len(fixations_before['behind_fixations']))

        fixations_after = trial_data[trial_data['Average_time_minCRT'].between(0, 5)]
        if fixations_after.empty:
            average_fixation_before = np.nan
        else:
            average_fixation_after = 1 - (
                    sum(fixations_after['behind_fixations']) / len(fixations_after['behind_fixations']))

        dict['fixations_before'].append(average_fixation_before)
        dict['fixations_after'].append(average_fixation_after)

    df_before_after_behind_55_45 = pd.DataFrame(dict)
    df_before_behind_55_45 = df_before_after_behind_55_45[['fixations_before', 'time_before']].dropna()
    df_after_behind_55_45 = df_before_after_behind_55_45[['fixations_after', 'time_after']].dropna()

    r2, p2 = stats.pearsonr(df_before_behind_55_45['time_before'], df_before_behind_55_45['fixations_before'])
    r3, p3 = stats.pearsonr(df_after_behind_55_45['time_after'], df_after_behind_55_45['fixations_after'])

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # fig.suptitle('Linear regression analysis before-after the CRT participant behind condition 2 (55-45 km/h)')
    # fig.text(0.05, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    # fig.text(0.5, 0.05, "Time [s]", ha="center", va="center")
    #
    # sns.regplot(df_before_behind_55_45, x='time_before', y='fixations_before', ax=axes[0])
    # sns.regplot(df_after_behind_55_45, x='time_after', y='fixations_after', ax=axes[1])
    #
    # axes[0].set(xlabel=None, ylabel=None)
    # axes[1].set(xlabel=None, ylabel=None)
    #
    # axes[0].set_title('5 seconds before the CRT')
    # axes[1].set_title('5 seconds after the CRT')
    #
    # axes[0].plot([], [], ' ', label='r: ' + str(round(r, 2)))
    # axes[0].plot([], [], ' ', label='p: ' + "{:.2e}".format(p))
    # axes[1].plot([], [], ' ', label='r: ' + str(round(r1, 2)))
    # axes[1].plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))
    #
    # axes[0].legend(loc='upper left')
    # axes[1].legend(loc='upper left')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    fig.suptitle('Linear regression analysis before-after the CRT condition 2 (55-45 km/h)')
    fig.text(0.05, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical', fontsize=12)
    fig.text(0.51, 0.48, "Participant is ahead", ha="center", va='center', fontsize=12)
    fig.text(0.51, 0.92, "Participant is behind", ha="center", va='center', fontsize=12)
    fig.text(0.51, 0.05, "Time [s]", ha="center", va="center", fontsize=12)

    plt.subplots_adjust(hspace=0.3)

    sns.regplot(df_before_ahead_55_45, x='time_before', y='fixations_before', ax=axes[0][0])
    sns.regplot(df_after_ahead_55_45, x='time_after', y='fixations_after', ax=axes[0][1])
    sns.regplot(df_before_behind_55_45, x='time_before', y='fixations_before', ax=axes[1][0])
    sns.regplot(df_after_behind_55_45, x='time_after', y='fixations_after', ax=axes[1][1])

    axes[0][0].set(xlabel=None, ylabel=None)
    axes[0][1].set(xlabel=None, ylabel=None)
    axes[1][0].set(xlabel=None, ylabel=None)
    axes[1][1].set(xlabel=None, ylabel=None)

    axes[0][0].set_title('5 seconds before the CRT')
    axes[0][1].set_title('5 seconds after the CRT')
    axes[1][0].set_title('5 seconds before the CRT')
    axes[1][1].set_title('5 seconds after the CRT')

    axes[0][0].plot([], [], ' ', label='r: ' + str(round(r, 2)))
    axes[0][0].plot([], [], ' ', label='p: ' + "{:.2e}".format(p))
    axes[0][1].plot([], [], ' ', label='r: ' + str(round(r1, 2)))
    axes[0][1].plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))
    axes[1][0].plot([], [], ' ', label='r: ' + str(round(r2, 2)))
    axes[1][0].plot([], [], ' ', label='p: ' + "{:.2e}".format(p2))
    axes[1][1].plot([], [], ' ', label='r: ' + str(round(r3, 2)))
    axes[1][1].plot([], [], ' ', label='p: ' + "{:.2e}".format(p3))

    # axes[0][0].set_ylim([0, 1])
    # axes[0][1].set_ylim([0, 1])
    # axes[1][0].set_ylim([0, 1])
    # axes[1][1].set_ylim([0, 1])

    axes[0][0].legend(loc='upper left')
    axes[0][1].legend(loc='upper left')
    axes[1][0].legend(loc='upper left')
    axes[1][1].legend(loc='upper left')

    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import os
import numpy as np
import datetime
from natsort import natsorted
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
    # --------------------------------------------------
    # 50-50
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'

    dict50_50 = plot_varjo(path_to_csv_50_50, '50-50', 'equal')

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict50_50.items()]))

    df['time_v1_new'] = df['time_vehicle1'] - df['CRT']
    df['time_v2_new'] = df['time_vehicle2'] - df['CRT']
    df['Average_time'] = df[["time_v1_new", "time_v2_new"]].mean(axis=1)
    df['Average_fixation'] = df[["gaze_vehicle1", "gaze_vehicle2"]].mean(axis=1)

    min_time_equal = min(min(df['time_v1_new']), min(df['time_v2_new']))
    max_time_equal = max(max(df['time_v1_new']), max(df['time_v2_new']))

    lengths = []
    for trial_number in df['trial'].unique():
        trial_data = df.loc[df['trial'] == trial_number, :]
        a = len(trial_data['Average_fixation'])
        lengths.append(a)

    max_length = max(lengths)
    new_time = np.linspace(min_time_equal, max_time_equal, max_length)

    new_df = {'trial': [],
              'new_time': [],
              'new_data': []}

    for trial_number in df['trial'].unique():
        trial_data = df.loc[df['trial'] == trial_number, :]
        new_df['new_time'] += list(new_time)
        new_df['new_data'] += list(trial_data['Average_fixation'])
        new_df['new_data'] += [np.nan] * (max_length - len(trial_data))
        new_df['trial'] += [trial_number] * max_length

    new_df = pd.DataFrame(new_df)


    fig, ax5 = plt.subplots(1, 1)
    fig.suptitle('Gaze behavior before-after the CRT condition 1 (50-50 km/h)')

    line_combined = sns.lineplot(x="new_time", y="new_data", data=new_df)

    l1 = line_combined.lines[0]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]


    ax5.fill_between(x1, y1, color='blue', alpha=0.1, label='Fixation on road')
    ax5.fill_between(x1, y1, 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax5.set_xlim([min_time_equal, max_time_equal])
    ax5.set_ylim([0, 1])

    ax5.set(xlabel='CRT [s]', ylabel='Fixation on opponent [%]')
    ax5.legend(loc='lower left')



    # # # ## -----------

    # 55_45
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    dict55_45_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '55-45', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    dict55_45_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '55-45', 'vehicle2')

    # df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v1_ahead.items()]))
    df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v2_ahead.items()]))

    # for ahead
    ahead_fixations_55_45 = pd.concat([df1['gaze_vehicle1'], df2['gaze_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_fixations')
    ahead_time_55_45 = pd.concat([df1['time_vehicle1'], df2['time_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_time')
    ahead_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0, ignore_index=True)
    ahead_trails = pd.concat([df1['trial'], df2['trial']], axis=0, ignore_index=True)
    df_ahead_55_45 = pd.concat([ahead_fixations_55_45, ahead_time_55_45, ahead_crt_55_45, ahead_trails],
                               axis=1)
    df_ahead_55_45['ahead_time'] = df_ahead_55_45['ahead_time'] - df_ahead_55_45['CRT']

    min_time_ahead = min(df_ahead_55_45['ahead_time'])
    max_time_ahead = max(df_ahead_55_45['ahead_time'])

    lengths = []
    for trial_number in df_ahead_55_45['trial'].unique():
        trial_data = df_ahead_55_45.loc[df_ahead_55_45['trial'] == trial_number, :]
        a = len(trial_data['ahead_fixations'])
        lengths.append(a)

    max_length = max(lengths)
    new_time = np.linspace(min_time_ahead, max_time_ahead, max_length)

    new_df_ahead = {'trial': [],
                    'new_time': [],
                    'new_data': []}

    for trial_number in df_ahead_55_45['trial'].unique():
        trial_data = df_ahead_55_45.loc[df_ahead_55_45['trial'] == trial_number, :]
        new_df_ahead['new_time'] += list(new_time)
        new_df_ahead['new_data'] += list(trial_data['ahead_fixations'])
        new_df_ahead['new_data'] += [np.nan] * (max_length - len(trial_data))
        new_df_ahead['trial'] += [trial_number] * max_length

    new_df_ahead = pd.DataFrame(new_df_ahead)

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
    df_behind_55_45['behind_time'] = df_behind_55_45['behind_time'] - df_behind_55_45['CRT']

    min_time_behind = min(df_behind_55_45['behind_time'])
    max_time_behind = max(df_behind_55_45['behind_time'])

    lengths = []
    for trial_number in df_behind_55_45['trial'].unique():
        trial_data = df_behind_55_45.loc[df_behind_55_45['trial'] == trial_number, :]
        a = len(trial_data['behind_fixations'])
        lengths.append(a)

    max_length = max(lengths)
    new_time = np.linspace(min_time_behind, max_time_behind, max_length)

    new_df_behind = {'trial': [],
                     'new_time': [],
                     'new_data': []}

    for trial_number in df_behind_55_45['trial'].unique():
        trial_data = df_behind_55_45.loc[df_behind_55_45['trial'] == trial_number, :]
        new_df_behind['new_time'] += list(new_time)
        new_df_behind['new_data'] += list(trial_data['behind_fixations'])
        new_df_behind['new_data'] += [np.nan] * (max_length - len(trial_data))
        new_df_behind['trial'] += [trial_number] * max_length

    new_df_behind = pd.DataFrame(new_df_behind)

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Gaze behavior before-after the CRT condition 2 (55-45 km/h)')
    fig.text(0.07, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    line_ahead = sns.lineplot(x="new_time", y="new_data", data=new_df_behind, ax=axes[0])
    line_behind = sns.lineplot(x="new_time", y="new_data", data=new_df_ahead, ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    l1 = line_ahead.lines[0]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    l2 = line_behind.lines[0]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]

    ysmoothed_1 = gaussian_filter1d(y1, sigma=4)
    ysmoothed_2 = gaussian_filter1d(y2, sigma=4)

    axes[0].plot(x1, ysmoothed_1, color = 'slateblue')
    axes[1].plot(x2, ysmoothed_2, color = 'slateblue')

    axes[0].fill_between(x1, ysmoothed_1, color='blue', alpha=0.1, label='Fixation on road')
    axes[0].fill_between(x1, ysmoothed_1, 1, color='red', alpha=0.1, label='Fixation on opponent')
    axes[1].fill_between(x2, ysmoothed_2, color='blue', alpha=0.1, label='Fixation on road')
    axes[1].fill_between(x2, ysmoothed_2, 1, color='red', alpha=0.1, label='Fixation on opponent')

    axes[0].set_xlim([min_time_behind, max_time_behind])
    axes[1].set_xlim([min_time_ahead, max_time_ahead])

    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])

    axes[0].set(xlabel=None, ylabel=None)
    axes[1].set(xlabel=None, ylabel=None)


    axes[0].legend(loc='lower left')
    axes[1].legend(loc='lower left')

    # # -----------------------------------------------------
    # 60-40
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'

    dict60_40_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '60-40', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'

    dict60_40_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '60-40', 'vehicle2')

    # df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict60_40_v1_ahead.items()]))
    df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict60_40_v2_ahead.items()]))

    # for ahead
    ahead_fixations_60_40 = pd.concat([df1['gaze_vehicle1'], df2['gaze_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_fixations')
    ahead_time_60_40 = pd.concat([df1['time_vehicle1'], df2['time_vehicle2']], axis=0, ignore_index=True).rename(
        'ahead_time')
    ahead_crt_60_40 = pd.concat([df1['CRT'], df2['CRT']], axis=0, ignore_index=True)
    ahead_trails_60_40 = pd.concat([df1['trial'], df2['trial']], axis=0, ignore_index=True)
    df_ahead_60_40 = pd.concat([ahead_fixations_60_40, ahead_time_60_40, ahead_crt_60_40, ahead_trails_60_40],
                               axis=1).dropna()
    df_ahead_60_40['ahead_time'] = df_ahead_60_40['ahead_time'] - df_ahead_60_40['CRT']

    min_time_ahead = min(df_ahead_60_40['ahead_time'])
    max_time_ahead = max(df_ahead_60_40['ahead_time'])

    lengths = []
    for trial_number in df_ahead_60_40['trial'].unique():
        trial_data = df_ahead_60_40.loc[df_ahead_60_40['trial'] == trial_number, :]
        a = len(trial_data['ahead_fixations'])
        lengths.append(a)

    max_length = max(lengths)
    new_time = np.linspace(min_time_ahead, max_time_ahead, max_length)

    new_df_ahead = {'trial': [],
                    'new_time': [],
                    'new_data': []}

    for trial_number in df_ahead_60_40['trial'].unique():
        trial_data = df_ahead_60_40.loc[df_ahead_60_40['trial'] == trial_number, :]
        new_df_ahead['new_time'] += list(new_time)
        new_df_ahead['new_data'] += list(trial_data['ahead_fixations'])
        new_df_ahead['new_data'] += [np.nan] * (max_length - len(trial_data))
        new_df_ahead['trial'] += [trial_number] * max_length

    new_df_ahead = pd.DataFrame(new_df_ahead)

    # --------------------------------------------------------------
    # for behind
    behind_fixations_60_40 = pd.concat([df1['gaze_vehicle2'], df2['gaze_vehicle1']], axis=0, ignore_index=True).rename(
        'behind_fixations')
    behind_time_60_40 = pd.concat([df1['time_vehicle2'], df2['time_vehicle1']], axis=0, ignore_index=True).rename(
        'behind_time')
    behind_crt_60_40 = pd.concat([df1['CRT'], df2['CRT']], axis=0, ignore_index=True)
    behind_trails = pd.concat([df1['trial'], df2['trial']], axis=0, ignore_index=True)
    df_behind_60_40 = pd.concat([behind_fixations_60_40, behind_time_60_40, behind_crt_60_40, behind_trails],
                                axis=1).dropna()
    df_behind_60_40['behind_time'] = df_behind_60_40['behind_time'] - df_behind_60_40['CRT']

    min_time_behind = min(df_behind_60_40['behind_time'])
    max_time_behind = max(df_behind_60_40['behind_time'])

    lengths = []
    for trial_number in df_behind_60_40['trial'].unique():
        trial_data = df_behind_60_40.loc[df_behind_60_40['trial'] == trial_number, :]
        a = len(trial_data['behind_fixations'])
        lengths.append(a)

    max_length = max(lengths)
    new_time = np.linspace(min_time_behind, max_time_behind, max_length)

    new_df_behind = {'trial': [],
                     'new_time': [],
                     'new_data': []}

    for trial_number in df_behind_60_40['trial'].unique():
        trial_data = df_behind_60_40.loc[df_behind_60_40['trial'] == trial_number, :]
        new_df_behind['new_time'] += list(new_time)
        new_df_behind['new_data'] += list(trial_data['behind_fixations'])
        new_df_behind['new_data'] += [np.nan] * (max_length - len(trial_data))
        new_df_behind['trial'] += [trial_number] * max_length

    new_df_behind = pd.DataFrame(new_df_behind)

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Gaze behavior before-after the CRT condition 3 (60-40 km/h)')
    fig.text(0.07, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    line_ahead = sns.lineplot(x="new_time", y="new_data", data=new_df_behind, ax=axes[0])
    line_behind = sns.lineplot(x="new_time", y="new_data", data=new_df_ahead, ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    l1 = line_ahead.lines[0]
    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]
    l2 = line_behind.lines[0]
    x2 = l2.get_xydata()[:, 0]
    y2 = l2.get_xydata()[:, 1]

    ysmoothed_1 = gaussian_filter1d(y1, sigma=4)
    ysmoothed_2 = gaussian_filter1d(y2, sigma=4)

    axes[0].plot(x1, ysmoothed_1, color = 'slateblue')
    axes[1].plot(x2, ysmoothed_2, color = 'slateblue')

    axes[0].fill_between(x1, ysmoothed_1, color='blue', alpha=0.1, label='Fixation on road')
    axes[0].fill_between(x1, ysmoothed_1, 1, color='red', alpha=0.1, label='Fixation on opponent')
    axes[1].fill_between(x2, ysmoothed_2, color='blue', alpha=0.1, label='Fixation on road')
    axes[1].fill_between(x2, ysmoothed_2, 1, color='red', alpha=0.1, label='Fixation on opponent')

    axes[0].set_xlim([min_time_behind, max_time_behind])
    axes[1].set_xlim([min_time_ahead, max_time_ahead])

    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])

    axes[0].set(xlabel=None, ylabel=None)
    axes[1].set(xlabel=None, ylabel=None)

    # axes[0].plot([], [], '', label='Average fixation before: ' + str(round(behind_average_fixation_before, 2)))
    # axes[0].plot([], [], '', label='Average fixation after: ' + str(round(behind_average_fixation_after, 2)))
    # axes[1].plot([], [], '', label='Average fixation before: ' + str(round(ahead_average_fixation_before, 2)))
    # axes[1].plot([], [], '', label='Average fixation after: ' + str(round(ahead_average_fixation_after, 2)))

    axes[0].legend(loc='lower left')
    axes[1].legend(loc='lower left')

    plt.show()

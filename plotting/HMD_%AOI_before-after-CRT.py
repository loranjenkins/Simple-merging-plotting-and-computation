import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import os
import numpy as np
import datetime
from matplotlib import pyplot
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


def interpolate(inp, fi):
    i, f = int(fi // 1), fi % 1  # Split floating-point index into whole & fractional parts.
    j = i + 1 if f > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


def average(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


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

    # dict = {'time_vehicle1': [], 'time_vehicle2': [], 'gaze_vehicle1': [], 'gaze_vehicle2': [], 'CRT': [], 'trail': []}
    dict = {'time_vehicle1': [], 'time_vehicle2': [], 'gaze_vehicle1': [], 'gaze_vehicle2': [], 'trail': [], 'CRT': []}

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

    path_to_data_csv = os.path.join('..', 'data_folder', 'crt_who_is_ahead.csv')
    global_crt_median = pd.read_csv(path_to_data_csv, sep=',')

    trails = 1
    for i in range(len(time_in_seconds_trails_v1)):
        inner = len(time_in_seconds_trails_v1[i])
        # print(inner)
        for i in range(inner):
            dict['trail'].append(trails)
        trails += 1

    if condition == '50-50':
        if who_ahead == 'equal':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt_median['crt_50_50'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

    if condition == '55-45':
        if who_ahead == 'vehicle1':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt_median['crt_45_55_vehicle1'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1
        elif who_ahead == 'vehicle2':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt_median['crt_55_45_vehicle2'][0 + index]
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

    if condition == '60-40':
        if who_ahead == 'vehicle1':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt_median['crt_40_60_vehicle1'][0 + index]
                # print(inner)
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1
        elif who_ahead == 'vehicle2':
            index = 0
            for i in range(len(time_in_seconds_trails_v1)):
                inner = len(time_in_seconds_trails_v1[i])
                first_crt = global_crt_median['crt_60_40_vehicle2'][0 + index]
                for i in range(inner):
                    dict['CRT'].append(first_crt)
                index += 1

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
    # 55_45
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    dict55_45_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '55-45', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    dict55_45_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '55-45', 'vehicle2')

    # df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v1_ahead.items()]))
    df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict55_45_v2_ahead.items()]))

    # for ahead
    ahead_fixations_55_45 = pd.concat([df1['gaze_vehicle1'], df2['gaze_vehicle2']], axis=0).rename(
        'ahead_fixations')
    ahead_time_55_45 = pd.concat([df1['time_vehicle1'], df2['time_vehicle2']], axis=0).rename(
        'ahead_time')
    ahead_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    ahead_trails = pd.concat([df1['trail'], df2['trail']], axis=0)
    df_ahead_55_45 = pd.concat([ahead_fixations_55_45, ahead_time_55_45, ahead_crt_55_45, ahead_trails],
                               axis=1).dropna()
    df_ahead_55_45['ahead_time'] = df_ahead_55_45['ahead_time'] - df_ahead_55_45['CRT']

    min_time = min(df_ahead_55_45['ahead_time'])
    max_time = max(df_ahead_55_45['ahead_time'])

    new_time = np.linspace(min_time, max_time, 2000)

    new_df_ahead = {'trial': [],
                    'new_time': [],
                    'new_data': []}

    for trial_number in df_ahead_55_45['trail'].unique():
        trial_data = df_ahead_55_45.loc[df_ahead_55_45['trail'] == trial_number, :]

        new_df_ahead['new_time'] += list(new_time)
        new_df_ahead['new_data'] += list(np.interp(new_time, trial_data['ahead_time'],
                                                   trial_data['ahead_fixations']))
        new_df_ahead['trial'] += [trial_number] * len(new_time)

    new_df_ahead = pd.DataFrame(new_df_ahead)

    ahead_data_before_crt = new_df_ahead[new_df_ahead['new_time'] < 0]
    ahead_on_road_fixation_before = list(ahead_data_before_crt['new_data']).count(1)
    ahead_on_opponent_fixation_before = list(ahead_data_before_crt['new_data']).count(0)

    ahead_average_fixation_before = ahead_on_opponent_fixation_before / sum(
        [ahead_on_road_fixation_before, ahead_on_opponent_fixation_before])

    ahead_data_after_crt = new_df_ahead[new_df_ahead['new_time'] > 0]
    ahead_on_road_fixation_after = list(ahead_data_after_crt['new_data']).count(1)
    ahead_on_opponent_fixation_after = list(ahead_data_after_crt['new_data']).count(0)

    ahead_average_fixation_after = ahead_on_opponent_fixation_after / sum(
        [ahead_on_road_fixation_after, ahead_on_opponent_fixation_after])

    # --------------------------------------------------------------
    # for behind
    behind_fixations_55_45 = pd.concat([df1['gaze_vehicle2'], df2['gaze_vehicle1']], axis=0).rename(
        'behind_fixations')
    behind_time_55_45 = pd.concat([df1['time_vehicle2'], df2['time_vehicle1']], axis=0).rename(
        'behind_time')
    behind_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    behind_trails = pd.concat([df1['trail'], df2['trail']], axis=0)
    df_behind_55_45 = pd.concat([behind_fixations_55_45, behind_time_55_45, behind_crt_55_45, behind_trails],
                                axis=1).dropna()
    df_behind_55_45['behind_time'] = df_behind_55_45['behind_time'] - df_behind_55_45['CRT']

    min_time = min(df_behind_55_45['behind_time'])
    max_time = max(df_behind_55_45['behind_time'])

    new_time = np.linspace(min_time, max_time, 2000)

    new_df_behind = {'trial': [],
                     'new_time': [],
                     'new_data': []}

    for trial_number in df_behind_55_45['trail'].unique():
        trial_data = df_behind_55_45.loc[df_behind_55_45['trail'] == trial_number, :]

        new_df_behind['new_time'] += list(new_time)
        new_df_behind['new_data'] += list(np.interp(new_time, trial_data['behind_time'],
                                                    trial_data['behind_fixations']))
        new_df_behind['trial'] += [trial_number] * len(new_time)

    new_df_behind = pd.DataFrame(new_df_behind)

    behind_data_before_crt = new_df_behind[new_df_behind['new_time'] < 0]
    behind_on_road_fixation_before = list(behind_data_before_crt['new_data']).count(1)
    behind_on_opponent_fixation_before = list(behind_data_before_crt['new_data']).count(0)

    behind_average_fixation_before = behind_on_opponent_fixation_before / sum(
        [behind_on_road_fixation_before, behind_on_opponent_fixation_before])

    behind_data_after_crt = new_df_behind[new_df_behind['new_time'] > 0]
    behind_on_road_fixation_after = list(behind_data_after_crt['new_data']).count(1)
    behind_on_opponent_fixation_after = list(behind_data_after_crt['new_data']).count(0)

    behind_average_fixation_after = behind_on_opponent_fixation_after / sum(
        [behind_on_road_fixation_after, behind_on_opponent_fixation_after])

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Gaze behavior before-after the CRT condition 2 (55-45 km/h)')
    fig.text(0.06, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    sns.lineplot(x="new_time", y="new_data", data=new_df_behind, ax=axes[0])
    sns.lineplot(x="new_time", y="new_data", data=new_df_ahead, ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    axes[0].set(xlabel=None, ylabel=None)
    axes[1].set(xlabel=None, ylabel=None)

    axes[0].plot([], [], '', label='Average fixation before: ' + str(round(behind_average_fixation_before, 2)))
    axes[0].plot([], [], '', label='Average fixation after: ' + str(round(behind_average_fixation_after, 2)))
    axes[1].plot([], [], '', label='Average fixation before: ' + str(round(ahead_average_fixation_before, 2)))
    axes[1].plot([], [], '', label='Average fixation after: ' + str(round(ahead_average_fixation_after, 2)))

    axes[0].legend(loc='lower left')
    axes[1].legend(loc='lower left')

    # -----------------------------------------------------
    # 60-40
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'

    dict60_40_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '60-40', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'

    dict60_40_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '60-40', 'vehicle2')

    # df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict60_40_v1_ahead.items()]))
    df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict60_40_v2_ahead.items()]))

    # for ahead
    ahead_fixations_60_40 = pd.concat([df1['gaze_vehicle1'], df2['gaze_vehicle2']], axis=0).rename(
        'ahead_fixations')
    ahead_time_60_40 = pd.concat([df1['time_vehicle1'], df2['time_vehicle2']], axis=0).rename(
        'ahead_time')
    ahead_crt_60_40 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    ahead_trails_60_40 = pd.concat([df1['trail'], df2['trail']], axis=0)
    df_ahead_60_40 = pd.concat([ahead_fixations_60_40, ahead_time_60_40, ahead_crt_60_40, ahead_trails_60_40],
                               axis=1).dropna()
    df_ahead_60_40['ahead_time'] = df_ahead_60_40['ahead_time'] - df_ahead_60_40['CRT']

    min_time = min(df_ahead_60_40['ahead_time'])
    max_time = max(df_ahead_60_40['ahead_time'])

    new_time = np.linspace(min_time, max_time, 2000)

    new_df_ahead = {'trial': [],
                    'new_time': [],
                    'new_data': []}

    for trial_number in df_ahead_60_40['trail'].unique():
        trial_data = df_ahead_60_40.loc[df_ahead_60_40['trail'] == trial_number, :]

        new_df_ahead['new_time'] += list(new_time)
        new_df_ahead['new_data'] += list(np.interp(new_time, trial_data['ahead_time'],
                                                   trial_data['ahead_fixations']))
        new_df_ahead['trial'] += [trial_number] * len(new_time)

    new_df_ahead = pd.DataFrame(new_df_ahead)

    ahead_data_before_crt = new_df_ahead[new_df_ahead['new_time'] < 0]
    ahead_on_road_fixation_before = list(ahead_data_before_crt['new_data']).count(1)
    ahead_on_opponent_fixation_before = list(ahead_data_before_crt['new_data']).count(0)

    ahead_average_fixation_before = ahead_on_opponent_fixation_before / sum(
        [ahead_on_road_fixation_before, ahead_on_opponent_fixation_before])

    ahead_data_after_crt = new_df_ahead[new_df_ahead['new_time'] > 0]
    ahead_on_road_fixation_after = list(ahead_data_after_crt['new_data']).count(1)
    ahead_on_opponent_fixation_after = list(ahead_data_after_crt['new_data']).count(0)

    ahead_average_fixation_after = ahead_on_opponent_fixation_after / sum(
        [ahead_on_road_fixation_after, ahead_on_opponent_fixation_after])

    # --------------------------------------------------------------
    # for behind
    behind_fixations_60_40 = pd.concat([df1['gaze_vehicle2'], df2['gaze_vehicle1']], axis=0).rename(
        'behind_fixations')
    behind_time_60_40 = pd.concat([df1['time_vehicle2'], df2['time_vehicle1']], axis=0).rename(
        'behind_time')
    behind_crt_60_40 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    behind_trails = pd.concat([df1['trail'], df2['trail']], axis=0)
    df_behind_60_40 = pd.concat([behind_fixations_60_40, behind_time_60_40, behind_crt_60_40, behind_trails],
                                axis=1).dropna()
    df_behind_60_40['behind_time'] = df_behind_60_40['behind_time'] - df_behind_60_40['CRT']

    min_time = min(df_behind_60_40['behind_time'])
    max_time = max(df_behind_60_40['behind_time'])

    new_time = np.linspace(min_time, max_time, 2000)

    new_df_behind = {'trial': [],
                     'new_time': [],
                     'new_data': []}

    for trial_number in df_behind_60_40['trail'].unique():
        trial_data = df_behind_60_40.loc[df_behind_60_40['trail'] == trial_number, :]

        new_df_behind['new_time'] += list(new_time)
        new_df_behind['new_data'] += list(np.interp(new_time, trial_data['behind_time'],
                                                    trial_data['behind_fixations']))
        new_df_behind['trial'] += [trial_number] * len(new_time)

    new_df_behind = pd.DataFrame(new_df_behind)

    behind_data_before_crt = new_df_behind[new_df_behind['new_time'] < 0]
    behind_on_road_fixation_before = list(behind_data_before_crt['new_data']).count(1)
    behind_on_opponent_fixation_before = list(behind_data_before_crt['new_data']).count(0)

    behind_average_fixation_before = behind_on_opponent_fixation_before / sum(
        [behind_on_road_fixation_before, behind_on_opponent_fixation_before])

    behind_data_after_crt = new_df_behind[new_df_behind['new_time'] > 0]
    behind_on_road_fixation_after = list(behind_data_after_crt['new_data']).count(1)
    behind_on_opponent_fixation_after = list(behind_data_after_crt['new_data']).count(0)

    behind_average_fixation_after = behind_on_opponent_fixation_after / sum(
        [behind_on_road_fixation_after, behind_on_opponent_fixation_after])

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Gaze behavior before-after the CRT condition 3 (60-40 km/h)')
    fig.text(0.06, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    sns.lineplot(x="new_time", y="new_data", data=new_df_behind, ax=axes[0])
    sns.lineplot(x="new_time", y="new_data", data=new_df_ahead, ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    axes[0].set(xlabel=None, ylabel=None)
    axes[1].set(xlabel=None, ylabel=None)

    axes[0].plot([], [], '', label='Average fixation before: ' + str(round(behind_average_fixation_before, 2)))
    axes[0].plot([], [], '', label='Average fixation after: ' + str(round(behind_average_fixation_after, 2)))
    axes[1].plot([], [], '', label='Average fixation before: ' + str(round(ahead_average_fixation_before, 2)))
    axes[1].plot([], [], '', label='Average fixation after: ' + str(round(ahead_average_fixation_after, 2)))

    axes[0].legend(loc='lower left')
    axes[1].legend(loc='lower left')

    # --------------------------------------------------
    # 50-50
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    dict50_50 = plot_varjo(path_to_csv_50_50, '50-50', 'equal')

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict50_50.items()]))

    df['time_v1_new'] = df['time_vehicle1'] - df['CRT']
    df['time_v2_new'] = df['time_vehicle2'] - df['CRT']
    df['Average_time'] = df[["time_v1_new", "time_v2_new"]].mean(axis=1)
    df['Average_fixation'] = df[["gaze_vehicle1", "gaze_vehicle2"]].mean(axis=1)

    min_time = min(min(df['time_v1_new']), min(df['time_v2_new']))
    max_time = max(max(df['time_v1_new']), max(df['time_v2_new']))

    new_time = np.linspace(min_time, max_time, 2000)

    new_df = {'trial': [],
              'new_time': [],
              'new_data': []}

    for trial_number in df['trail'].unique():
        trial_data = df.loc[df['trail'] == trial_number, :]

        new_df['new_time'] += list(new_time)
        new_df['new_data'] += list(np.interp(new_time, trial_data['Average_time'],
                                             trial_data['Average_fixation']))
        new_df['trial'] += [trial_number] * len(new_time)

    new_df = pd.DataFrame(new_df)

    data_before_crt = new_df[new_df['new_time'] < 0]
    on_road_fixation_before = list(data_before_crt['new_data']).count(1)
    both_else_fixation_before = list(data_before_crt['new_data']).count(0.5)
    on_opponent_fixation_before = list(data_before_crt['new_data']).count(0)

    average_fixation_before = (on_opponent_fixation_before + both_else_fixation_before / 2) / sum(
        [on_road_fixation_before, both_else_fixation_before, on_opponent_fixation_before])

    data_after_crt = new_df[new_df['new_time'] > 0]
    on_road_fixation_after = list(data_after_crt['new_data']).count(1)
    both_else_fixation_after = list(data_after_crt['new_data']).count(0.5)
    on_opponent_fixation_after = list(data_after_crt['new_data']).count(0)

    average_fixation_after = (on_opponent_fixation_after + both_else_fixation_after / 2) / sum(
        [on_road_fixation_after, both_else_fixation_after, on_opponent_fixation_after])

    fig, ax5 = plt.subplots(1, 1)
    fig.suptitle('Gaze behavior before-after the CRT condition 1 (50-50 km/h)')
    sns.lineplot(x="new_time", y="new_data", data=new_df)
    ax5.plot([], [], ' ', label='Average % before: ' + str(round(average_fixation_before, 2)))
    ax5.plot([], [], ' ', label='Average % after: ' + str(round(average_fixation_after, 2)))
    ax5.set(xlabel='CRT [s]', ylabel='Fixation on opponent [%]')
    ax5.legend(loc='lower left')

    plt.show()

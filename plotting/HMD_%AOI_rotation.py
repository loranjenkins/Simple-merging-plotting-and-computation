import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path

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



# def plot_varjo_data(list_of_csv):
#     simulation_constants = SimulationConstants(vehicle_width=2,
#                                                vehicle_length=4.7,
#                                                tunnel_length=118,
#                                                track_width=8,
#                                                track_height=195,
#                                                track_start_point_distance=390,
#                                                track_section_length_before=275.77164466275354,
#                                                track_section_length_after=200)  # goes until 400
#
#     track = SymmetricMergingTrack(simulation_constants)
#
#     all_pds_list = []
#     for i in range(len(list_of_csv)):
#         all_pds = pd.read_csv(trails[i], sep=',')
#         all_pds_list.append(all_pds)
#
#     xy_coordinates_for_trace = []
#     for i in range(1):
#         xy_coordinates = vehicle_xy_coordinates(2, all_pds_list[i])
#         # xy_coordinates = [list(a) for a in zip(xy_coordinates_for_trace[0], xy_coordinates_for_trace[1])]
#         # xy_coordinates_for_trace.append(xy_coordinates)
#         print(xy_coordinates)


if __name__ == '__main__':
    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=118,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=200)  # goes until 400
    track = SymmetricMergingTrack(simulation_constants)

    files_directory = r'C:\Users\loran\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined\Test varjo'
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)


    all_pds_list = []
    for i in range(len(trails)):
        all_pds = pd.read_csv(trails[i], sep=',')
        all_pds_list.append(all_pds)



    xy_coordinates_for_trace = []
    for i in range(len(all_pds_list)):
        xy_coordinates_vehicle = vehicle_xy_coordinates(2, all_pds_list[i])
        xy_coordinates_zip = [list(a) for a in zip(xy_coordinates_vehicle[0], xy_coordinates_vehicle[1])]
        xy_coordinates_for_trace.append(xy_coordinates_zip)

    # print(xy_coordinates_for_trace[0])


    travelled_distance = []
    for list_index in range(len(xy_coordinates_for_trace)):
        individual_xy_vehicle = xy_coordinates_for_trace[list_index]
        inner_xy_list = []
        for xy in range(len(individual_xy_vehicle)):
            traveled_distance_vehicle = track.coordinates_to_traveled_distance(individual_xy_vehicle[xy])
            inner_xy_list.append(traveled_distance_vehicle)
        travelled_distance.append(inner_xy_list)




# if __name__ == '__main__':
#     files_directory = r'C:\Users\loran\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined\Test varjo'
#     trails = []
#     for file in Path(files_directory).glob('*.csv'):
#         # trail_condition = plot_trail(file)
#         trails.append(file)
#
#     simulation_constants = SimulationConstants(vehicle_width=2,
#                                                vehicle_length=4.7,
#                                                tunnel_length=118,
#                                                track_width=8,
#                                                track_height=195,
#                                                track_start_point_distance=390,
#                                                track_section_length_before=275.77164466275354,
#                                                track_section_length_after=200)  # goes until 400
#
#     track = SymmetricMergingTrack(simulation_constants)
#
#     data = pd.read_csv(trails[0], sep=',')
#     data1 = pd.read_csv(trails[1], sep=',')
#
#     xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
#     xy_coordinates_vehicle1_trail1 = vehicle_xy_coordinates(2, data1)
#
#     xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
#     xy_coordinates_vehicle1_trail1 = [list(a) for a in zip(xy_coordinates_vehicle1_trail1[0], xy_coordinates_vehicle1_trail1[1])]
#
#     travelled_distance_vehicle1 = []
#     travelled_distance_vehicle1_trail1 = []
#     for i in range(len(xy_coordinates_vehicle1)):
#         traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
#         traveled_distance1_trail1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1_trail1[i])
#         travelled_distance_vehicle1.append(traveled_distance1)
#         travelled_distance_vehicle1_trail1.append(traveled_distance1_trail1)
#
#     index_of_tunnel_vehicle1 = min(range(len(travelled_distance_vehicle1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1[i] - track.tunnel_length))
#
#     index_of_mergepoint_vehicle1 = min(range(len(travelled_distance_vehicle1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1[i] - simulation_constants.track_section_length_before))
#
#     index_of_tunnel_vehicle1_trail1 = min(range(len(travelled_distance_vehicle1_trail1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1_trail1[i] - track.tunnel_length))
#
#     index_of_mergepoint_vehicle1_trail1 = min(range(len(travelled_distance_vehicle1_trail1)),
#                                        key=lambda i: abs(travelled_distance_vehicle1_trail1[
#                                                              i] - simulation_constants.track_section_length_before))
#
#     interactive_area_trace = travelled_distance_vehicle1[index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1]
#     interactive_area_trace_trail1 = travelled_distance_vehicle1_trail1[index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1]
#
#     hmd_interactive_area = list(data['HMD_rotation_vehicle1'][index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1])
#     hmd_interactive_area_trail1 = list(data1['HMD_rotation_vehicle1'][index_of_tunnel_vehicle1_trail1:index_of_mergepoint_vehicle1_trail1])
#
#     interactive_dict = {'interactive_area_trace': [], 'hmd_interactive_area': []}
#     interactive_dict_trail1 = {'interactive_area_trace': [], 'hmd_interactive_area': []}
#
#     for i in range(len(interactive_area_trace)):
#         interactive_dict['interactive_area_trace'].append(interactive_area_trace[i])
#         interactive_dict['hmd_interactive_area'].append(hmd_interactive_area[i])
#
#     for i in range(len(interactive_area_trace_trail1)):
#         interactive_dict_trail1['interactive_area_trace'].append(interactive_area_trace_trail1[i])
#         interactive_dict_trail1['hmd_interactive_area'].append(hmd_interactive_area_trail1[i])
#
#     df = pd.DataFrame(interactive_dict)
#     df1 = pd.DataFrame(interactive_dict_trail1)
#
#     on_ramp_vs_opponent = []
#     for i in range(len(df['hmd_interactive_area'])):
#         if df['hmd_interactive_area'][i] > 0.98:
#             on_ramp_vs_opponent.append(1)
#         elif df['hmd_interactive_area'][i] < 0.98:
#             on_ramp_vs_opponent.append(0)
#
#     on_ramp_vs_opponent_trail1 = []
#     for i in range(len(df1['hmd_interactive_area'])):
#         if df1['hmd_interactive_area'][i] > 0.98:
#             on_ramp_vs_opponent_trail1.append(1)
#         elif df1['hmd_interactive_area'][i] < 0.98:
#             on_ramp_vs_opponent_trail1.append(0)
#
#     df['on_ramp_vs_opponent'] = on_ramp_vs_opponent
#     df1['on_ramp_vs_opponent'] = on_ramp_vs_opponent_trail1
#
#     df_travelled_trace_combined = pd.concat([df['interactive_area_trace'], df1['interactive_area_trace']], axis=1, keys=['x1', 'x2'])
#     df_travelled_trace_combined['mean'] = df_travelled_trace_combined.mean(axis=1)
#
#     df_fixation_combined = pd.concat([df['on_ramp_vs_opponent'], df1['on_ramp_vs_opponent']], axis=1, keys=['y1', 'y2'])
#     df_fixation_combined['mean'] = df_fixation_combined.mean(axis=1)
#
#
#     fig, (ax1) = plt.subplots(1)
#     # fig.suptitle('-')
#     # plt.title('positions over time')
#     ax1.set_xlabel('average travelled distance [m]')
#     ax1.set_ylabel('% fixated on AOI')
#     # ax1.set_yticks([175, -5, 5, -175])
#     # ax1.set_yticklabels([175, 0, 0, -175])
#
#
#     # ax1.set_xlim(50, 450)
#
#     ax1.plot(df_travelled_trace_combined['mean'], df_fixation_combined['mean'])
#
#     plt.show()




























#
#
#
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import datetime
# from pathlib import Path
#
# from trackobjects.simulationconstants import SimulationConstants
# from trackobjects.symmetricmerge import SymmetricMergingTrack
#
# def vehicle_xy_coordinates(intcolumn, data_csv):
#     list_x = []
#     list_y = []
#     for i in range(len(data_csv.iloc[:, intcolumn])):
#         transform_vehicle1 = eval(data_csv.iloc[i, intcolumn])
#         x_loc = transform_vehicle1[0]
#         y_loc = transform_vehicle1[1]
#
#         list_x.append(x_loc)
#         # list_x = list(dict.fromkeys(list_x))
#         list_y.append(y_loc)
#         # list_y = list(dict.fromkeys(list_y))
#
#     return list_x, list_y
#
# if __name__ == '__main__':
#     files_directory = r'C:\Users\loran\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined\Test varjo'
#     trails = []
#     for file in Path(files_directory).glob('*.csv'):
#         # trail_condition = plot_trail(file)
#         trails.append(file)
#
#     simulation_constants = SimulationConstants(vehicle_width=2,
#                                                vehicle_length=4.7,
#                                                tunnel_length=118,
#                                                track_width=8,
#                                                track_height=195,
#                                                track_start_point_distance=390,
#                                                track_section_length_before=275.77164466275354,
#                                                track_section_length_after=200)  # goes until 400
#
#     track = SymmetricMergingTrack(simulation_constants)
#
#     data = pd.read_csv(trails[0], sep=',')
#     data1 = pd.read_csv(trails[1], sep=',')
#
#     xy_coordinates_vehicle1 = vehicle_xy_coordinates(2, data)
#     xy_coordinates_vehicle1_trail1 = vehicle_xy_coordinates(2, data1)
#
#     xy_coordinates_vehicle1 = [list(a) for a in zip(xy_coordinates_vehicle1[0], xy_coordinates_vehicle1[1])]
#     xy_coordinates_vehicle1_trail1 = [list(a) for a in zip(xy_coordinates_vehicle1_trail1[0], xy_coordinates_vehicle1_trail1[1])]
#
#     travelled_distance_vehicle1 = []
#     travelled_distance_vehicle1_trail1 = []
#     for i in range(len(xy_coordinates_vehicle1)):
#         traveled_distance1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1[i])
#         traveled_distance1_trail1 = track.coordinates_to_traveled_distance(xy_coordinates_vehicle1_trail1[i])
#         travelled_distance_vehicle1.append(traveled_distance1)
#         travelled_distance_vehicle1_trail1.append(traveled_distance1_trail1)
#
#     index_of_tunnel_vehicle1 = min(range(len(travelled_distance_vehicle1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1[i] - track.tunnel_length))
#
#     index_of_mergepoint_vehicle1 = min(range(len(travelled_distance_vehicle1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1[i] - simulation_constants.track_section_length_before))
#
#     index_of_tunnel_vehicle1_trail1 = min(range(len(travelled_distance_vehicle1_trail1)),
#                                    key=lambda i: abs(travelled_distance_vehicle1_trail1[i] - track.tunnel_length))
#
#     index_of_mergepoint_vehicle1_trail1 = min(range(len(travelled_distance_vehicle1_trail1)),
#                                        key=lambda i: abs(travelled_distance_vehicle1_trail1[
#                                                              i] - simulation_constants.track_section_length_before))
#
#     interactive_area_trace = travelled_distance_vehicle1[index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1]
#     interactive_area_trace_trail1 = travelled_distance_vehicle1_trail1[index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1]
#
#     hmd_interactive_area = list(data['HMD_rotation_vehicle1'][index_of_tunnel_vehicle1:index_of_mergepoint_vehicle1])
#     hmd_interactive_area_trail1 = list(data1['HMD_rotation_vehicle1'][index_of_tunnel_vehicle1_trail1:index_of_mergepoint_vehicle1_trail1])
#
#     interactive_dict = {'interactive_area_trace': [], 'hmd_interactive_area': []}
#     interactive_dict_trail1 = {'interactive_area_trace': [], 'hmd_interactive_area': []}
#
#     for i in range(len(interactive_area_trace)):
#         interactive_dict['interactive_area_trace'].append(interactive_area_trace[i])
#         interactive_dict['hmd_interactive_area'].append(hmd_interactive_area[i])
#
#     for i in range(len(interactive_area_trace_trail1)):
#         interactive_dict_trail1['interactive_area_trace'].append(interactive_area_trace_trail1[i])
#         interactive_dict_trail1['hmd_interactive_area'].append(hmd_interactive_area_trail1[i])
#
#     df = pd.DataFrame(interactive_dict)
#     df1 = pd.DataFrame(interactive_dict_trail1)
#
#     on_ramp_vs_opponent = []
#     for i in range(len(df['hmd_interactive_area'])):
#         if df['hmd_interactive_area'][i] > 0.98:
#             on_ramp_vs_opponent.append(1)
#         elif df['hmd_interactive_area'][i] < 0.98:
#             on_ramp_vs_opponent.append(0)
#
#     on_ramp_vs_opponent_trail1 = []
#     for i in range(len(df1['hmd_interactive_area'])):
#         if df1['hmd_interactive_area'][i] > 0.98:
#             on_ramp_vs_opponent_trail1.append(1)
#         elif df1['hmd_interactive_area'][i] < 0.98:
#             on_ramp_vs_opponent_trail1.append(0)
#
#     df['on_ramp_vs_opponent'] = on_ramp_vs_opponent
#     df1['on_ramp_vs_opponent'] = on_ramp_vs_opponent_trail1
#
#     df_travelled_trace_combined = pd.concat([df['interactive_area_trace'], df1['interactive_area_trace']], axis=1, keys=['x1', 'x2'])
#     df_travelled_trace_combined['mean'] = df_travelled_trace_combined.mean(axis=1)
#
#     df_fixation_combined = pd.concat([df['on_ramp_vs_opponent'], df1['on_ramp_vs_opponent']], axis=1, keys=['y1', 'y2'])
#     df_fixation_combined['mean'] = df_fixation_combined.mean(axis=1)
#
#
#     fig, (ax1) = plt.subplots(1)
#     # fig.suptitle('-')
#     # plt.title('positions over time')
#     ax1.set_xlabel('average travelled distance [m]')
#     ax1.set_ylabel('% fixated on AOI')
#     # ax1.set_yticks([175, -5, 5, -175])
#     # ax1.set_yticklabels([175, 0, 0, -175])
#
#
#     # ax1.set_xlim(50, 450)
#
#     ax1.plot(df_travelled_trace_combined['mean'], df_fixation_combined['mean'])
#
#     plt.show()



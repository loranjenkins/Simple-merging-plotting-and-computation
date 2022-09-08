import pickle
from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge_old import SymmetricMergingTrack
import numpy as np




if __name__ == '__main__':
    a_file = open("global_data_dict.pkl", "rb")
    data_dict = pickle.load(a_file)

    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=150)

    track = SymmetricMergingTrack(simulation_constants)

    average_travelled_distance_trace = list((np.array(data_dict['distance_traveled_vehicle1']) + np.array(
        data_dict['distance_traveled_vehicle2'])) / 2.)

    headway = np.array(data_dict['distance_traveled_vehicle1']) - np.array(data_dict['distance_traveled_vehicle2'])

    for value in range(len(average_travelled_distance_trace)):
        track.get_headway_bounds(average_travelled_distance_trace[value], simulation_constants.vehicle_width, simulation_constants.vehicle_length)

    # fig_1 = plt.figure()
    # raw_headway_axes = fig_1.add_subplot(1, 1, 1)
    # raw_headway_axes.set_aspect('equal')

    #See conflict_signal but u need get_headwaybounds first


    # raw_headway_axes.vlines([50, 100], 20, -20, linestyles='dashed', colors='lightgray', zorder=0.)
    # raw_headway_axes.text(25., 11., 'Tunnel', verticalalignment='center', horizontalalignment='center', clip_on=True)
    # raw_headway_axes.text(75., 11., 'Approach', verticalalignment='center', horizontalalignment='center', clip_on=True)
    # raw_headway_axes.text(125., 11., 'Car Following', verticalalignment='center', horizontalalignment='center', clip_on=True)

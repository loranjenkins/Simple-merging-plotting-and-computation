"""
Copyright 2022, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of simple-merging-experiment.

simple-merging-experiment is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

simple-merging-experiment is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with simple-merging-experiment.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import pickle

import tqdm
from matplotlib import pyplot as plt
import numpy as np

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack


def generate_data_dict():
    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=5,
                                               tunnel_length=118,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=150)  # goes until 400

    track = SymmetricMergingTrack(simulation_constants)
    end_point = track.total_distance

    data_dict = {'positive_headway_bound': [],
                 'negative_headway_bound': [],
                 'average_travelled_distance': []}

    for average_y_position_in_mm in tqdm.trange(int(end_point * 1000)):
        average_y_position = average_y_position_in_mm / 1000.

        lb, ub = track.get_headway_bounds(average_y_position,
                                          vehicle_length=simulation_constants.vehicle_length,
                                          vehicle_width=simulation_constants.vehicle_width) #if we switch length with width it plots

        data_dict['positive_headway_bound'].append(ub)
        data_dict['negative_headway_bound'].append(lb)
        data_dict['average_travelled_distance'].append(average_y_position)

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key], dtype=float)
    return data_dict


if __name__ == '__main__':

    path_to_saved_dict = os.path.join('..', 'data_folder', 'headway_bounds.pkl')
    if not os.path.isfile(path_to_saved_dict):
        data_dict = generate_data_dict()

        with open(path_to_saved_dict, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open(path_to_saved_dict, 'rb') as f:
            data_dict = pickle.load(f)

    plt.figure()
    plt.plot(data_dict['average_travelled_distance'], data_dict['positive_headway_bound'])
    plt.plot(data_dict['average_travelled_distance'], data_dict['negative_headway_bound'])

    plt.show()
import autograd.numpy as np
import matplotlib.pyplot as plt
import shapely.affinity
import shapely.geometry
import shapely.ops
import math

from scipy import stats

from trackobjects.track import Track
from trackobjects.trackside import TrackSide


class SymmetricMergingTrack(Track):
    def __init__(self, simulation_constants):
        self._start_point_distance = simulation_constants.track_start_point_distance
        self._section_length_before = simulation_constants.track_section_length_before
        self._section_length_after = simulation_constants.track_section_length_after
        self._track_width = simulation_constants.track_width
        self._track_height = simulation_constants.track_height
        self._approach_angle = math.atan(self.track_width/self._track_height)
        if not np.pi / 4 > self._approach_angle > 0:
            raise ValueError('The approach angle for the symmetric merging track cannot be larger then 45 degree, please decrease the start point distance or '
                             'increase the section length.')

        # self._merge_point = np.array([0.0, np.sqrt(self._section_length ** 2 - (self._start_point_distance / 2) ** 2)])
        self._merge_point = np.array([0.0, -np.sqrt(self._section_length_before ** 2 - (self._start_point_distance / 2) ** 2)]) #start point was from y = 10 instead of 7.5 -> fix to generic
        self._end_point = np.array([0.0, self._merge_point[1] - self._section_length_after])

        self._left_way_points = [np.array([-self._start_point_distance / 2., -7.5]), self._merge_point, self._end_point]
        self._left_run_up_point = np.array([-self._start_point_distance, -np.sqrt(self._section_length_before ** 2 - (self._start_point_distance / 2) ** 2)])
        self._right_way_points = [np.array([self._start_point_distance / 2., -7.5]), self._merge_point, self._end_point]
        self._right_run_up_point = np.array([self._start_point_distance, -np.sqrt(self._section_length_before ** 2 - (self._start_point_distance / 2) ** 2)])

        self._lower_bound_threshold = None
        self._upper_bound_threshold = None

        self._upper_bound_approximation_slope = None
        self._upper_bound_approximation_intersect = None
        self._lower_bound_approximation_slope = None
        self._lower_bound_approximation_intersect = None
        self._lower_bound_constant_value = None

        if type(self) == SymmetricMergingTrack:
            # only initialize the approximation when type is SymmetricMergingTrack to prevent this initialization to be called in a super().__init__() call
            self._initialize_linear_bound_approximation(simulation_constants.vehicle_width, simulation_constants.vehicle_length)

    def _initialize_linear_bound_approximation(self, vehicle_width, vehicle_length):
        self._upper_bound_threshold = self._section_length_before - (vehicle_width / 2.) / np.tan((np.pi / 2) - self._approach_angle) - (vehicle_length / 2)
        self._lower_bound_threshold = self._section_length_before - (vehicle_width / 2.) / np.tan((np.pi / 2) - self._approach_angle) + (vehicle_length / 2)

        if self._lower_bound_threshold > self._section_length_before:
            self._lower_bound_threshold = self._section_length_before

        last_point = 2 * self._section_length_before

        # 10 cm resolution lookup
        entries = [i for i in range(int(self._upper_bound_threshold * 10 + 1), int(last_point * 10))]

        look_up_table = np.zeros((len(entries), 2))

        for index in range(len(entries)):
            travelled_distance = entries[index] / 10.
            look_up_table[index, :] = self.get_collision_bounds(travelled_distance, vehicle_width, vehicle_length)

        self._upper_bound_approximation_slope, self._upper_bound_approximation_intersect, _, _, _ = stats.linregress(np.array(entries) / 10., look_up_table[:, 1])
        lower_bound_index = []
        for i in range(len(entries)):
            comparision = np.where(entries[i] > self._lower_bound_threshold * 10)
            lower_bound_index.append(comparision)

        # self._lower_bound_approximation_slope, self._lower_bound_approximation_intersect, _, _, _ = stats.linregress(np.array(entries[lower_bound_index:]) / 10.,
        #                                                                                                            look_up_table[lower_bound_index:, 0])

        self._lower_bound_constant_value, _ = self.get_collision_bounds(self._lower_bound_threshold - 0.1, vehicle_width, vehicle_length)

    def is_beyond_track_bounds(self, position):
        _, distance_to_track = self.closest_point_on_route(position)
        return distance_to_track > self.track_width / 2.0

    def is_beyond_finish(self, position):
        return position[1] >= self._end_point[1]

    def get_heading(self, position):
        """
        Assumes that approach angle is <45 degrees and not 0 degrees.
        With these assumption, the relevant section can be determined based on the y coordinates only
        """
        if position[1] > self._merge_point[1]:
            # closest point is on final section
            return np.pi / 2
        else:
            if position[0] > 0.0:
                # closest point is on right approach
                return np.pi - self._approach_angle
            else:
                # closest point is on left approach
                return self._approach_angle

    def closest_point_on_route_rightvehicle(self, position):
        """
        Assumes that approach angle is <45 degrees and not 0 degrees.
        With these assumption, the relevant section can be determined based on the y coordinates only and the approach can be expressed as y = ax + b
        """

        if position[1] < self._merge_point[1]:
            before_or_after = 'after'
            track_side = None
        else:
            before_or_after = 'before'

        return self._closest_point_on_route_forced_rightvehicle(position , before_or_after)

    def _closest_point_on_route_forced_rightvehicle(self, position, before_or_after_merge):
        closest_point_on_route, shortest_distance = None, None
        if before_or_after_merge == 'after':
            closest_point_on_route = np.array([self._end_point[0], position[1]])
            shortest_distance = abs(position[0] - self._end_point[0])
        elif before_or_after_merge == 'before':
            # define the approach line as y = ax + b and use the formula found here:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula

            x0, y0 = self._right_way_points[0]

            x1, y1 = self._merge_point
            b = y1
            a = - (b / x0)

            x = (position[0] + a * position[1] - a * b) / (a ** 2 + 1)
            y = a * x + b
            closest_point_on_route = np.array([x, y])
            shortest_distance = abs(b + a * position[0] - position[1]) / np.sqrt(a ** 2 + 1)
        return closest_point_on_route, shortest_distance

    def closest_point_on_route_leftvehicle(self, position):
        """
        Assumes that approach angle is <45 degrees and not 0 degrees.
        With these assumption, the relevant section can be determined based on the y coordinates only and the approach can be expressed as y = ax + b
        """

        if position[1] < self._merge_point[1]:
            before_or_after = 'after'
            track_side = None
        else:
            before_or_after = 'before'

        return self._closest_point_on_route_forced_leftvehicle(position, before_or_after)

    def _closest_point_on_route_forced_leftvehicle(self, position, before_or_after_merge):
        closest_point_on_route, shortest_distance = None, None
        if before_or_after_merge == 'after':
            closest_point_on_route = np.array([self._end_point[0], position[1]])
            shortest_distance = abs(position[0] - self._end_point[0])
        elif before_or_after_merge == 'before':
            # define the approach line as y = ax + b and use the formula found here:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula

            x0, y0 = self._left_way_points[0]

            x1, y1 = self._merge_point
            b = y1
            a = - (b / x0)

            x = (position[0] + a * position[1] - a * b) / (a ** 2 + 1)
            y = a * x + b
            closest_point_on_route = np.array([x, y])
            shortest_distance = abs(b + a * position[0] - position[1]) / np.sqrt(a ** 2 + 1)
        return closest_point_on_route, shortest_distance

    def traveled_distance_to_coordinates_rightvehicle(self, distance):
        if distance <= self._section_length_before:
            before_or_after = 'before'
        else:
            before_or_after = 'after'

        return self._traveled_distance_to_coordinates_forced_rightvehicle(distance, before_or_after_merge=before_or_after)

    def _traveled_distance_to_coordinates_forced_rightvehicle(self, distance, before_or_after_merge):
        if before_or_after_merge == 'before':
            x = ((self._start_point_distance / 2.) - np.cos(self._approach_angle) * distance)
            y = -np.sin(self._approach_angle) * distance
        elif before_or_after_merge == 'after':
            x = 0.0
            y = -np.sin(self._approach_angle) * self._section_length_after + (distance - self._section_length_after)
        return np.array([x, y])

    def traveled_distance_to_coordinates_leftvehicle(self, distance):
        if distance <= self._section_length_before:
            before_or_after = 'before'
        else:
            before_or_after = 'after'

        return self._traveled_distance_to_coordinates_forced_leftvehicle(distance, before_or_after_merge=before_or_after)

    def _traveled_distance_to_coordinates_forced_leftvehicle(self, distance, before_or_after_merge):
        if before_or_after_merge == 'before':
            x = ((self._start_point_distance / 2.) - np.cos(self._approach_angle) * distance) * -1
            y = -np.sin(self._approach_angle) * distance
        elif before_or_after_merge == 'after':
            x = 0.0
            y = -np.sin(self._approach_angle) * self._section_length_after + (distance - self._section_length_after)
        return np.array([x, y])

    def coordinates_to_traveled_distance(self, point):
        if point[0] == 0.0:
            before_or_after = 'after'
        elif point[0] > 0.0:
            before_or_after = 'before'
        elif point[0] < 0.0:
            before_or_after = 'before'
        return self._coordinates_to_traveled_distance_forced(point, before_or_after_merge=before_or_after)

    def _coordinates_to_traveled_distance_forced(self, point, before_or_after_merge):
        if before_or_after_merge == 'after':
            distance = self._section_length_after + (point[1] - self._merge_point[1])
        elif before_or_after_merge == 'before':
            if point[0] < 0:
                distance = np.linalg.norm(point - self._left_way_points[0])
            elif point[0] > 0:
                distance = np.linalg.norm(point - self._right_way_points[0])
        return distance

    def get_collision_bounds_approximation(self, traveled_distance_vehicle_1):
        if traveled_distance_vehicle_1 < self._upper_bound_threshold:
            return None, None

        else:
            ub = self._upper_bound_approximation_slope * traveled_distance_vehicle_1 + self._upper_bound_approximation_intersect
            if traveled_distance_vehicle_1 > self._lower_bound_threshold:
                lb = self._lower_bound_approximation_slope * traveled_distance_vehicle_1 + self._lower_bound_approximation_intersect
            else:
                lb = self._lower_bound_constant_value

            return lb, ub

    def get_collision_bounds(self, traveled_distance_vehicle_1, vehicle_width, vehicle_length):
        """
        Returns the bounds on the position of the other vehicle that spans the set of all collision positions. Assumes both vehicles have the same dimensions.
        returns (None, None) when no collisions are possible

        :param traveled_distance_vehicle_1:
        :param vehicle_width:
        :param vehicle_length:
        :return:
        """

        # setup path_polygon and other pre-requisites
        a = self._approach_angle
        b = np.pi / 2 - self._approach_angle
        l = vehicle_length / 2
        w = vehicle_width / 2

        straight_part = shapely.geometry.box(-w, self._merge_point[1] - l, w, self._merge_point[1] - self._section_length_after - l)
        # fig, (ax1, ax2) = plt.subplots(2)
        # x, y = straight_part.exterior.xy
        # ax1.plot(y, x)

        R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])

        top_left = R @ np.array([-w, l]) + self._merge_point
        top_right = R @ np.array([w, l]) + self._merge_point

        start_point_right = self.traveled_distance_to_coordinates_rightvehicle(-7.5)

        bottom_left = R @ np.array([-w, -l]) + start_point_right
        bottom_right = R @ np.array([w, -l]) + start_point_right

        approach_part = shapely.geometry.Polygon([top_left, top_right, bottom_right, bottom_left])
        # x, y = approach_part.exterior.xy
        # ax2.plot(y, x)
        # ax2.invert_xaxis()
        # ax2.invert_yaxis()
        # plt.show()
        # setup polygon representing vehicle 1
        vehicle_1 = shapely.geometry.box(-w, -l, w, l)

        if traveled_distance_vehicle_1 <= self._section_length_before:
            vehicle_1 = shapely.affinity.rotate(vehicle_1, -b, use_radians=True)

        vehicle_1_position = self.traveled_distance_to_coordinates_leftvehicle(traveled_distance_vehicle_1)
        vehicle_1 = shapely.affinity.translate(vehicle_1, -vehicle_1_position[0], -vehicle_1_position[1])


        # get intersection between polygons
        straight_intersection = straight_part.intersection(vehicle_1)
        approach_intersection = approach_part.intersection(vehicle_1)

        if straight_intersection.is_empty and approach_intersection.is_empty:
            return None, None
        else:
            s_lower_bounds = []
            s_upper_bounds = []
            a_lower_bounds = []
            a_upper_bounds = []

            if not straight_intersection.is_empty:
                exterior_points_straight = np.array(straight_intersection.exterior.coords.xy).T
                for point in exterior_points_straight:
                    lb, ub, _ = self._get_straight_bounds_for_point(point, l)
                    s_lower_bounds += [lb]
                    s_upper_bounds += [ub]

            if not approach_intersection.is_empty:
                exterior_points_approach = np.array(approach_intersection.exterior.coords.xy).T
                for point in exterior_points_approach:
                    lb, ub, _ = self._get_approach_bounds_for_point(point, l)
                    a_lower_bounds += [lb]
                    a_upper_bounds += [ub]

            upper_bounds = s_upper_bounds + a_upper_bounds
            lower_bounds = s_lower_bounds + a_lower_bounds

            upper_bounds = [b for b in upper_bounds if not np.isnan(b)]
            lower_bounds = [b for b in lower_bounds if not np.isnan(b)]

            return min(lower_bounds, default=0), max(upper_bounds, default=0)

    def _get_straight_bounds_for_point(self, point, l):
        closest_point_on_route_after_merge, _ = self._closest_point_on_route_forced_rightvehicle(point, before_or_after_merge='after')
        traveled_distance_after_merge = self._coordinates_to_traveled_distance_forced(closest_point_on_route_after_merge,
                                                                                      before_or_after_merge='after')

        after_merge_lb = traveled_distance_after_merge - l
        after_merge_ub = traveled_distance_after_merge + l

        if after_merge_lb < self._section_length_after:
            after_merge_lb = self._section_length_after
        if after_merge_ub < self._section_length_after:
            after_merge_ub = np.nan

        return after_merge_lb, after_merge_ub, closest_point_on_route_after_merge

    def _get_approach_bounds_for_point(self, point, l):
        closest_point_on_route_before_merge, _ = self._closest_point_on_route_forced_rightvehicle(point, before_or_after_merge='before')
        traveled_distance_before_merge = self._coordinates_to_traveled_distance_forced(closest_point_on_route_before_merge,
                                                                                       before_or_after_merge='before')

        before_merge_lb = traveled_distance_before_merge - l
        before_merge_ub = traveled_distance_before_merge + l

        if before_merge_lb > self._section_length_before:
            before_merge_lb = np.nan
        if before_merge_ub > self._section_length_before:
            before_merge_ub = self._section_length_before

        return before_merge_lb, before_merge_ub, closest_point_on_route_before_merge

    def get_track_bounding_rect(self):
        x1 = self._left_way_points[0][0]
        x2 = self._right_way_points[0][0]

        y1 = 0.0
        y2 = self._end_point[1]

        return x1, y1, x2, y2

    def get_way_points(self, track_side: TrackSide, show_run_up=False) -> list:
        if track_side is TrackSide.LEFT:
            if show_run_up:
                return [self._left_run_up_point] + self._left_way_points
            else:
                return self._left_way_points
        else:
            if show_run_up:
                return [self._right_run_up_point] + self._right_way_points
            else:
                return self._right_way_points

    def get_start_position(self, track_side: TrackSide) -> np.ndarray:
        if track_side is TrackSide.LEFT:
            return self._left_way_points[0]
        else:
            return self._right_way_points[0]

    @property
    def total_distance(self) -> float:
        return self._section_length * 2.

    @property
    def track_width(self) -> float:
        return self._track_width

    @staticmethod
    def _plot_polygons(polygons: list, points=None):
        """
        For debugging of the collision bounds

        :param polygons:
        :param points:
        :return:
        """
        from matplotlib import pyplot
        from descartes import PolygonPatch

        fig = pyplot.figure()
        ax = fig.add_subplot(111)

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        color_index = 0

        for polygon in polygons:
            ax.add_patch(PolygonPatch(polygon, fc=colors[color_index], alpha=0.5))
            color_index += 1
            if color_index >= len(colors):
                color_index = 0

        if points is not None:
            for points_set in points:
                pyplot.scatter(x=points_set[:, 0], y=points_set[:, 1])

        ax.autoscale(enable=True)
        pyplot.show()

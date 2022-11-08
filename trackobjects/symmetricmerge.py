import autograd.numpy as np
import matplotlib.pyplot as plt
import shapely.affinity
import shapely.geometry
import shapely.ops
import math
from scipy import optimize


class SymmetricMergingTrack:
    def __init__(self, simulation_constants):
        self.start_point_distance = simulation_constants.track_start_point_distance
        self.section_length_before = simulation_constants.track_section_length_before
        self.section_length_after = simulation_constants.track_section_length_after
        self.tunnel_length = simulation_constants.tunnel_length
        self.track_width = simulation_constants.track_width
        self.track_height = simulation_constants.track_height
        self.approach_angle = math.atan((self.start_point_distance / 2) / self.track_height)

        self.merge_point = np.array([0.0, 230])
        self.end_point = np.array([0.0, self.merge_point[1] + self.section_length_after])

        self.left_way_points = [np.array([-self.start_point_distance / 2., 0]), self.merge_point, self.end_point]
        self.right_way_points = [np.array([self.start_point_distance / 2., 0]), self.merge_point, self.end_point]

        self.lower_bound_threshold = None
        self.upper_bound_threshold = None

        self.initialize_linear_bound_approximation(simulation_constants.vehicle_width,
                                                    simulation_constants.vehicle_length)

    def initialize_linear_bound_approximation(self, vehicle_width, vehicle_length):
        self.upper_bound_threshold = self.section_length_before - (vehicle_width / 2.) / np.tan(
            (np.pi / 2) - self.approach_angle) - (vehicle_length / 2)
        self.lower_bound_threshold = self.section_length_before - (vehicle_width / 2.) / np.tan(
            (np.pi / 2) - self.approach_angle) + (vehicle_length / 2)
        if self.lower_bound_threshold > self.section_length_before:
            self.lower_bound_threshold = self.section_length_before

    def get_headway_bounds(self, average_travelled_distance, vehicle_width, vehicle_length):
            """
            Returns the bounds on the headway that spans the set of all collision positions. Assumes both vehicles have the same dimensions.
            returns (None, None) when no collisions are possible.

            This method uses scipy optimize to find the minimal headway without a collision, this is inefficient but this method is only used for plotting purposes.
            In the simulations, please use the get_collision_bounds method, it has a closed form solution.

            :param average_travelled_distance:
            :param vehicle_width:
            :param vehicle_length:
            :return:
            """

            if average_travelled_distance > self.section_length_before + vehicle_length / 2.:
                # both vehicles are on the straight section
                return -vehicle_length/2, vehicle_length/2
            elif average_travelled_distance < self.upper_bound_threshold:
                # at least one of the vehicles is on the approach on a position where it cannot collide
                return None, None
            else:
                # find the minimal headway (x) where the overlap between the vehicles is negative (no collision) and x is positive
                solution = optimize.minimize(lambda x: abs(x), np.array([0.]), constraints=[{'type': 'ineq',
                                                                                             'fun': self._collision_constraint,
                                                                                             'args': (
                                                                                             average_travelled_distance,
                                                                                             vehicle_width,
                                                                                             vehicle_length)},
                                                                                            {'type': 'ineq',
                                                                                             'fun': lambda x: x}])
                headway = solution.x[0]
                # the headway bounds are completely symmetrical
                return -headway/2, headway/2

    def _collision_constraint(self, head_way, average_travelled_distance, vehicle_width, vehicle_length):
        left = average_travelled_distance + head_way / 2.
        right = average_travelled_distance - head_way / 2.

        lb, ub = self.get_collision_bounds(left, vehicle_width, vehicle_length)

        if lb is None:
            return 1.
        else:
            return lb - right

    def position_is_beyond_track_bounds(self, position):
        _, distance_to_track = self.closest_point_on_route(position)
        return distance_to_track > self.track_width / 2.0

    def position_is_beyond_finish(self, position):
        return position[1] >= self.end_point[1]

    def get_heading(self, position):
        """
        Assumes that approach angle is <45 degrees and not 0 degrees.
        With these assumption, the relevant section can be determined based on the y coordinates only
        """
        if position[1] > self.merge_point[1]:
            # closest point is on final section
            return np.pi / 2
        else:
            if position[0] > 0.0:
                # closest point is on right approach
                return np.pi - self.approach_angle
            else:
                # closest point is on left approach
                return self.approach_angle

    def closest_point_on_route(self, position):
        """
        Assumes that approach angle is <45 degrees and not 0 degrees.
        With these assumption, the relevant section can be determined based on the y coordinates only and the approach can be expressed as y = ax + b
        """

        if position[1] > self.merge_point[1]:
            before_or_after = 'after'
            left_or_right = None
        else:
            before_or_after = 'before'

            if position[0] >= 0.0:
                # closest point is on right approach
                left_or_right = 'right'
            else:
                # closest point is on left approach
                left_or_right = 'left'

        return self._closest_point_on_route_forced(position, left_or_right, before_or_after)

    def _closest_point_on_route_forced(self, position, left_or_right, before_or_after_merge):
        closest_point_on_route, shortest_distance = None, None
        if before_or_after_merge == 'after':
            closest_point_on_route = np.array([self.end_point[0], position[1]])
            shortest_distance = abs(position[0] - self.end_point[0])
        elif before_or_after_merge == 'before':
            # define the approach line as y = ax + b and use the formula found here:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula

            if left_or_right == 'right':
                x0, y0 = self.right_way_points[0]
            elif left_or_right == 'left':
                x0, y0 = self.left_way_points[0]

            x1, y1 = self.merge_point
            b = y1
            a = - (b / x0)

            x = (position[0] + a * position[1] - a * b) / (a ** 2 + 1)
            y = a * x + b
            closest_point_on_route = np.array([x, y])
            shortest_distance = abs(b + a * position[0] - position[1]) / np.sqrt(a ** 2 + 1)
        return closest_point_on_route, shortest_distance

    def traveled_distance_to_coordinates(self, distance, vehicle='left'):
        if distance <= self.section_length_before:
            before_or_after = 'before'
        else:
            before_or_after = 'after'

        return self._traveled_distance_to_coordinates_forced(distance, left_or_right=vehicle,
                                                             before_or_after_merge=before_or_after)

    def _traveled_distance_to_coordinates_forced(self, distance, left_or_right, before_or_after_merge):
        if left_or_right == 'left':
            x_axis = -1
        elif left_or_right == 'right':
            x_axis = 1

        if before_or_after_merge == 'before':
            x = ((self.start_point_distance / 2.) - np.cos(self.approach_angle) * distance) * x_axis
            y = np.sin(self.approach_angle) * distance + self.left_way_points[0][1]
        elif before_or_after_merge == 'after':
            x = 0.0
            y = np.sin(self.approach_angle) * self.section_length_before + self.left_way_points[0][1] + (distance - self.section_length_before)
        return np.array([x, y])

    def coordinates_to_traveled_distance(self, point):
        if point[0] == 0.0:
            before_or_after = 'after'
            left_or_right = None
        elif point[0] > 0.0:
            before_or_after = 'before'
            left_or_right = 'right'
        elif point[0] < 0.0:
            before_or_after = 'before'
            left_or_right = 'left'
        return self._coordinates_to_traveled_distance_forced(point, left_or_right=left_or_right,
                                                             before_or_after_merge=before_or_after)

    def _coordinates_to_traveled_distance_forced(self, point, left_or_right, before_or_after_merge):
        if before_or_after_merge == 'after':
            distance = self.section_length_before + (point[1] - self.merge_point[1])
        elif before_or_after_merge == 'before':
            if left_or_right == 'left':
                distance = np.linalg.norm(point - self.left_way_points[0])
            elif left_or_right == 'right':
                distance = np.linalg.norm(point - self.right_way_points[0])
        return distance

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
        a = self.approach_angle
        b = np.pi / 2 - self.approach_angle
        l = vehicle_length / 2
        w = vehicle_width / 2

        straight_part = shapely.geometry.box(-w, self.merge_point[1] - l, w,
                                             self.merge_point[1] + self.section_length_after + l)

        # straight_part = shapely.geometry.box(-w, 235 - l, w,
        #                                      235 + self.section_length_after + l)

        # R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
        R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]]) #with y down positive
        # https://stackoverflow.com/questions/24675945/rotating-a-matrix-in-a-non-standard-2d-plane

        top_left = R @ np.array([-w, l]) + self.merge_point
        top_right = R @ np.array([w, l]) + self.merge_point

        # top_left = R @ np.array([-w, l]) + 235
        # top_right = R @ np.array([w, l]) + 235

        start_point_right = self.traveled_distance_to_coordinates(0, vehicle='right')

        bottom_left = R @ np.array([-w, -l]) + start_point_right
        bottom_right = R @ np.array([w, -l]) + start_point_right

        approach_part = shapely.geometry.Polygon([top_left, top_right, bottom_right, bottom_left])

        # setup polygon representing vehicle 1
        vehicle_1 = shapely.geometry.box(-w, -l, w, l)

        if traveled_distance_vehicle_1 <= self.section_length_before:
            vehicle_1 = shapely.affinity.rotate(vehicle_1, -b, use_radians=True)

        vehicle_1_position = self.traveled_distance_to_coordinates(traveled_distance_vehicle_1)

        vehicle_1 = shapely.affinity.translate(vehicle_1, vehicle_1_position[0], vehicle_1_position[1])
        # plt.plot(*straight_part.exterior.xy)
        # plt.plot(*approach_part.exterior.xy)
        # plt.plot(*vehicle_1.exterior.xy)
        # plt.show()
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
            return min(lower_bounds), max(upper_bounds)

    def _get_straight_bounds_for_point(self, point, l):
        closest_point_on_route_after_merge, _ = self._closest_point_on_route_forced(point, left_or_right='right',
                                                                                    before_or_after_merge='after')
        traveled_distance_after_merge = self._coordinates_to_traveled_distance_forced(
            closest_point_on_route_after_merge, left_or_right='right',
            before_or_after_merge='after')

        after_merge_lb = traveled_distance_after_merge - l
        after_merge_ub = traveled_distance_after_merge + l

        if after_merge_lb < self.section_length_before:
            after_merge_lb = self.section_length_before
        if after_merge_ub < self.section_length_before:
            after_merge_ub = np.nan

        return after_merge_lb, after_merge_ub, closest_point_on_route_after_merge

    def _get_approach_bounds_for_point(self, point, l):
        closest_point_on_route_before_merge, _ = self._closest_point_on_route_forced(point, left_or_right='right',
                                                                                     before_or_after_merge='before')
        traveled_distance_before_merge = self._coordinates_to_traveled_distance_forced(
            closest_point_on_route_before_merge, left_or_right='right',
            before_or_after_merge='before')

        before_merge_lb = traveled_distance_before_merge - l
        before_merge_ub = traveled_distance_before_merge + l

        if before_merge_lb > self.section_length_before:
            before_merge_lb = np.nan
        if before_merge_ub > self.section_length_before:
            before_merge_ub = self.section_length_before

        return before_merge_lb, before_merge_ub, closest_point_on_route_before_merge


    @property
    def total_distance(self) -> float:
        return self.section_length_before + self.section_length_after
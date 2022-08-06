class SimulationConstants:
    """ object that stores all constants needed to recall a saved simulation. """

    def __init__(self, vehicle_width, vehicle_length, track_width, track_height, track_start_point_distance, track_section_length_before, track_section_length_after):
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.track_width = track_width
        self.track_height = track_height
        self.track_start_point_distance = track_start_point_distance
        self.track_section_length_before = track_section_length_before
        self.track_section_length_after = track_section_length_after

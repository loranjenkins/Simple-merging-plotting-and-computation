def check_if_on_collision_course_for_point(travelled_distance_collision_point, data):
    point_predictions = {}

    for side in TrackSide:
        point_predictions[side] = np.array(data['travelled_distance'][side]) + np.array(data['velocities'][side]) * (
                travelled_distance_collision_point - np.array(data['travelled_distance'][side.other])) / np.array(data['velocities'][side.other])

    lb, ub = data['track'].get_collision_bounds(travelled_distance_collision_point, data['vehicle_width'], data['vehicle_length'])

    on_collision_course = ((lb < point_predictions[TrackSide.LEFT]) & (point_predictions[TrackSide.LEFT] < ub)) | \
                          ((lb < point_predictions[TrackSide.RIGHT]) & (point_predictions[TrackSide.RIGHT] < ub))
    return on_collision_course

def calculate_conflict_resolved_time(data):
    time = [t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))]
    track = data['track']

    merge_point_collision_course = check_if_on_collision_course_for_point(2 * data['simulation_constants'].track_section_length, data)
    threshold_collision_course = check_if_on_collision_course_for_point(track._upper_bound_threshold + 1e-3, data)

    on_collision_course = merge_point_collision_course | threshold_collision_course

    approach_mask = ((np.array(data['travelled_distance'][TrackSide.RIGHT]) > data['simulation_constants'].track_section_length) &
                     (np.array(data['travelled_distance'][TrackSide.RIGHT]) < 2 * data['simulation_constants'].track_section_length)) | \
                    ((np.array(data['travelled_distance'][TrackSide.LEFT]) > data['simulation_constants'].track_section_length) &
                     (np.array(data['travelled_distance'][TrackSide.LEFT]) < 2 * data['simulation_constants'].track_section_length))

    indices_of_conflict_resolved = ((on_collision_course == False) & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved][0]
    except IndexError:
        time_of_conflict_resolved = None

    return time_of_conflict_resolved
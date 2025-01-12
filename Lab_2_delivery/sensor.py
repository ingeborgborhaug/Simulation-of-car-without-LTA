import config
import numpy as np
import random
import pygame
import window
import config

def get_car_bound_status(car_front_left, car_front_right, LTA_left_threshold, LTA_right_threshold, tolerance=config.LTA_TOLERANCE):
    """
    Check whether the car is inside the road, near the LTA bounds, or outside them.

    Args:
        car_front_left (tuple): (x, y) coordinates of the car's front-left corner.
        car_front_right (tuple): (x, y) coordinates of the car's front-right corner.
        LTA_left_points (list): List of (x, y) points representing the left LTA boundary.
        LTA_right_points (list): List of (x, y) points representing the right LTA boundary.
        tolerance (float): Distance threshold to consider the car near the boundaries.

    Returns:
        str: Status of the car relative to the road:
            - "inside": Car is well inside the road boundaries.
            - "near_left": Car is near the left boundary.
            - "near_right": Car is near the right boundary.
            - "outside_left": Car is outside the left boundary.
            - "outside_right": Car is outside the right boundary.
    """
    # Find the closest left and right LTA points
    closest_left_point = min(LTA_left_threshold, key=lambda point: np.sqrt((car_front_left[0] - point[0])**2 + (car_front_left[1] - point[1])**2))
    closest_right_point = min(LTA_right_threshold, key=lambda point: np.sqrt((car_front_right[0] - point[0])**2 + (car_front_right[1] - point[1])**2))

    # Check if car is on or crossing the LTA-threshold (red line)
    if car_front_left[0] <= closest_left_point[0]:
        return "crossing_LTA_threshold_left"
    if car_front_right[0] >= closest_right_point[0]:
        return "crossing_LTA_threshold_right"

    # Check if car is near the LTA-threshold (red line)
    left_distance = window.calculate_closest_distance(car_front_left[0], car_front_left[1], LTA_left_threshold)
    right_distance = window.calculate_closest_distance(car_front_right[0], car_front_right[1], LTA_right_threshold)

    if left_distance <= tolerance:
        return "near_LTA_threshold_left"
    if right_distance <= tolerance:
        return "near_LTA_threshold_right"

    # If none of the above, car is well inside
    return "inside"

def LTA_sensor(middle_line, right_line):
    
    # Compute LTA detections of middle and right line
    middle_line_raw = [(x, y) for x, y in middle_line]
    right_line_raw = [(x, y) for x, y in right_line]
    
    road_lines_dict = {
        "middle_line": middle_line_raw,
        "right_line": right_line_raw,
    }
    
    LTA_dict_noisy = {}
    
    # Noise parameters
    noise_std = config.NOISE_STD  # Standard deviation for Gaussian noise
    
    for key, points in road_lines_dict.items():
        sensor_output_line = []

        for x, y in points:
            if key in ["middle_line", "right_line"]:
                # Apply horizontal Gaussian noise
                x += np.random.normal(0, noise_std)
                sensor_output_line.append((x, y))

        LTA_dict_noisy[key] = sensor_output_line  
       
    return LTA_dict_noisy["middle_line"], LTA_dict_noisy["right_line"]

def LTA_processing(car_position_screen, LTA_left_points_noisy, LTA_right_points_noisy):
    """Estimate gaps in roadlines in middle line and right line, filling gaps inside the sensor cone."""
    LTA_left_points_processed = []
    LTA_right_points_processed = []
    LTA_left_estimated_indexes = []
    LTA_right_estimated_indexes = []

    assert len(LTA_left_points_noisy) >= 2 and len(LTA_right_points_noisy) >= 2  # Assume at least the two first points are good
    assert len(LTA_left_points_noisy) == len(LTA_right_points_noisy)

    x1, y1 = LTA_left_points_noisy[0]
    x2, y2 = LTA_left_points_noisy[1]
    initial_angle = np.arctan2(y2 - y1, x2 - x1)
    road_angles = [initial_angle]

    estimated_width = abs(LTA_right_points_noisy[0][0] - LTA_left_points_noisy[0][0])

    for i in range(1, len(LTA_left_points_noisy)):
        left = LTA_left_points_noisy[i]
        right = LTA_right_points_noisy[i]

        if not np.isnan(left).any() and not np.isnan(right).any():
            LTA_left_points_processed.append(left)
            LTA_right_points_processed.append(right)
            current_angle = -np.arctan2(left[1] - LTA_left_points_noisy[i - 1][1], left[0] - LTA_left_points_noisy[i - 1][0])
            road_angles.append(current_angle)
            estimated_width = abs(right[0] - left[0])

        elif np.isnan(left).any() and not np.isnan(right).any():
            current_angle = -np.arctan2(right[1] - LTA_right_points_noisy[i - 1][1], right[0] - LTA_right_points_noisy[i - 1][0])
            road_angles.append(current_angle)

            estimated_left_x = right[0] - estimated_width
            estimated_left_y = right[1]
            LTA_left_points_processed.append((estimated_left_x, estimated_left_y))
            LTA_right_points_processed.append(right)
            LTA_left_estimated_indexes.append(i)

        elif not np.isnan(left).any() and np.isnan(right).any():
            current_angle = -np.arctan2(left[1] - LTA_left_points_noisy[i - 1][1], left[0] - LTA_left_points_noisy[i - 1][0])
            road_angles.append(current_angle)

            estimated_right_x = left[0] + estimated_width
            estimated_right_y = left[1]
            LTA_left_points_processed.append(left)
            LTA_right_points_processed.append((estimated_right_x, estimated_right_y))
            LTA_right_estimated_indexes.append(i)

        else:
            # Both points are missing, estimate both based on previous points
            prev_left = LTA_left_points_processed[-1]
            prev_right = LTA_right_points_processed[-1]
            estimated_left_x = prev_left[0]
            estimated_right_x = prev_right[0]
            estimated_y = prev_left[1] + config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL / len(LTA_left_points_noisy)
            LTA_left_points_processed.append((estimated_left_x, estimated_y))
            LTA_right_points_processed.append((estimated_right_x, estimated_y))
            LTA_left_estimated_indexes.append(i)
            LTA_right_estimated_indexes.append(i)


    # Truncate LTA points inside the sensor lookahead
    def truncate_points(points):
        return [
            (x, y) for x, y in points
            if config.CAR_POSITION_SCREEN_INIT[1] - config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL <= y <= config.SCREEN_HEIGHT
        ]

    # Truncate estimated indexes inside the sensor lookahead
    def truncate_indexes(points, indexes):
        return [
            i for i in indexes
            if i < len(points) and config.CAR_POSITION_SCREEN_INIT[1] - config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL <= points[i][1] <= config.SCREEN_HEIGHT
        ]
    
    LTA_left_points_processed = truncate_points(LTA_left_points_processed)
    LTA_right_points_processed = truncate_points(LTA_right_points_processed)
    
    LTA_left_estimated_indexes = truncate_indexes(LTA_left_points_processed, LTA_left_estimated_indexes)
    LTA_right_estimated_indexes = truncate_indexes(LTA_right_points_processed, LTA_right_estimated_indexes)

    return (
        LTA_left_points_processed,
        LTA_right_points_processed,
        LTA_left_estimated_indexes,
        LTA_right_estimated_indexes,
    )

def draw_LTA_detections(screen, LTA_left_detection_processed, LTA_right_detection_processed, LTA_left_holes_indexes, LTA_right_holes_indexes):
    """
    Draw LTA left and right lines with red lines for detected points
    and blue lines for estimated points (previously np.nan).
    
    Args:
        screen: Pygame screen object.
        road_data: Precomputed road data containing lane points.
        LTA_left_points_processed: Processed left LTA points.
        LTA_right_points_processed: Processed right LTA points.
        LTA_left_outliers_indexes: Indexes of estimated left LTA points.
        LTA_right_outliers_indexes: Indexes of estimated right LTA points.
    """
    # Draw LTA left line
    for i in range(len(LTA_left_detection_processed) - 1):
        # color = config.CYAN if i in LTA_left_holes_indexes else config.YELLOW
        pygame.draw.line(screen, config.CYAN, LTA_left_detection_processed[i], LTA_left_detection_processed[i + 1], 2)

    # Draw LTA right line
    for i in range(len(LTA_right_detection_processed) - 1):
        # color = config.CYAN if i in LTA_right_holes_indexes else config.WHITE
        pygame.draw.line(screen, config.CYAN, LTA_right_detection_processed[i], LTA_right_detection_processed[i + 1], 2)

def generate_LTA_threshold(LTA_left_detection_processed, LTA_right_detection_processed):
    # LTA THRESHOLD
    LTA_left_threshold = [(x + config.LTA_TOLERANCE, y) for x, y in LTA_left_detection_processed]
    LTA_right_threshold = [(x - config.LTA_TOLERANCE, y) for x, y in LTA_right_detection_processed]
    
    LTA_threshold_dict = {
        "LTA_left": LTA_left_threshold,
        "LTA_right": LTA_right_threshold,
    }
    
    for key, points in LTA_threshold_dict.items():
        shortened_LTA = []
        
        # Truncate LTA lines to be INSIDE the sensor lookahead
        if key in ["LTA_left", "LTA_right"]:
            shortened_LTA = [
                (x, y) for x, y in points
                if config.CAR_POSITION_SCREEN_INIT[1] - config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL <= y <= config.SCREEN_HEIGHT
            ]
        
        LTA_threshold_dict[key] = shortened_LTA 
    
    return LTA_threshold_dict["LTA_left"], LTA_threshold_dict["LTA_right"]
            
def draw_LTA_threshold(screen, LTA_left_threshold, LTA_right_threshold):
    """
    Draw LTA left and right lines with red lines for detected points
    and blue lines for estimated points (previously np.nan).
    
    Args:
        screen: Pygame screen object.
        road_data: Precomputed road data containing lane points.
        LTA_left_points_processed: Processed left LTA points.
        LTA_right_points_processed: Processed right LTA points.
        LTA_left_outliers_indexes: Indexes of estimated left LTA points.
        LTA_right_outliers_indexes: Indexes of estimated right LTA points.
    """
    # Draw LTA left line
    for i in range(len(LTA_left_threshold) - 1):
        pygame.draw.line(screen, config.RED, LTA_left_threshold[i], LTA_left_threshold[i + 1], 2)

    # Draw LTA right line
    for i in range(len(LTA_right_threshold) - 1):
        pygame.draw.line(screen, config.RED, LTA_right_threshold[i], LTA_right_threshold[i + 1], 2)

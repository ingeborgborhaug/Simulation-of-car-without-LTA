import config
import pygame
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def make_holes_in_roadline(points, nan_probability=config.NAN_PROBABILITY, chunk_size_range=(5, 20)):
    """
    Replace random chunks of points with np.nan to simulate sensor dropouts.

    Args:
        points: List of (x, y) points to process.
        nan_probability: Probability of starting a missing chunk.
        chunk_size_range: Range of chunk sizes to remove (min, max).

    Returns:
        List of (x, y) points with random chunks replaced by (np.nan, np.nan).
    """
    points = points.copy()  # Avoid modifying the original list
    indexes = []
    
    i = 0
    while i < len(points):
        if i > 1 and random.random() < nan_probability:
            # Determine the size of the chunk to replace
            chunk_size = random.randint(*chunk_size_range)
            for j in range(chunk_size):
                if i + j < len(points):
                    points[i + j] = (np.nan, np.nan)
                    indexes.append(i+j)
            i += chunk_size  # Skip over the chunk
        else:
            i += 1

    return points, indexes

def generate_road_with_holes_in_roadlines():
    road = build_road()

    left_line = road["left"]
    middle_line = road["middle"]
    right_line = road["right"]

    left_line_with_holes, left_line_holes_indexes = make_holes_in_roadline(left_line)
    middle_line_with_holes, middle_line_holes_indexes = make_holes_in_roadline(middle_line)
    right_line_with_holes, right_line_holes_indexes = make_holes_in_roadline(right_line)

    road_with_holes = {
        "left": left_line_with_holes,
        "middle": middle_line_with_holes,
        "right": right_line_with_holes,
    }
    
    road_with_holes_indexes = {
        "left_indexes": left_line_holes_indexes,
        "middle_indexes": middle_line_holes_indexes,
        "right_indexes": right_line_holes_indexes,
    }
    
    return road_with_holes, road_with_holes_indexes

def draw_road_holes(screen, road_data, left_line_holes_indexes, middle_line_holes_indexes, right_line_holes_indexes):
    """
    Draws the entire road on the screen based on precomputed road data.
    Holes are represented by gaps in the lines by skipping segments where holes occur.

    Args:
        screen: Pygame screen object for drawing.
        road_data (dict): Precomputed road data containing lane points (with offset applied).
        left_line_holes_indexes, middle_line_holes_indexes, right_line_holes_indexes: Indexes of the points that should be gaps.
    """
    # Draw the right lane line (white), skipping segments with holes
    for i in range(len(road_data["right"]) - 1):
        if i not in right_line_holes_indexes and i + 1 not in right_line_holes_indexes:
            pygame.draw.line(screen, config.WHITE, road_data["right"][i], road_data["right"][i + 1], 4)

    # Draw the left lane line (white), skipping segments with holes
    for i in range(len(road_data["left"]) - 1):
        if i not in left_line_holes_indexes and i + 1 not in left_line_holes_indexes:
            pygame.draw.line(screen, config.WHITE, road_data["left"][i], road_data["left"][i + 1], 4)

    # Draw the middle lane line (yellow, dashed), skipping segments with holes
    for i in range(0, len(road_data["middle"]) - 1, 2):  # Dashed pattern
        if i not in middle_line_holes_indexes and i + 1 not in middle_line_holes_indexes:
            pygame.draw.line(screen, config.WHITE, road_data["middle"][i], road_data["middle"][i + 1], 3)


def straight_line_road(length, start_x, start_y, direction):
    """
    Generate a straight road segment.

    Args:
        length (float): Length of the road in meters.
        start_x (float): Starting x-coordinate in pixels.
        start_y (float): Starting y-coordinate in pixels.
        direction (float): Direction angle in radians (0 = vertical down).

    Returns:
        dict: Points for the left, middle, and right lanes.
    """
    # Convert length to pixels
    length_pixels = length * config.METER_TO_PIXEL

    # Generate points
    left_line_points = []
    middle_line_points = []
    right_line_points = []

    for offset in range(0, int(length_pixels), 10):  # Step size = 10 pixels
        dx = offset * np.sin(direction)  # Horizontal offset
        dy = -offset * np.cos(direction)  # Vertical offset (negative for upward)

        left_line_points.append((float(start_x - config.LANE_WIDTH / 2), float(start_y + dy)))
        middle_line_points.append((float(start_x), float(start_y + dy)))
        right_line_points.append((float(start_x + config.LANE_WIDTH / 2), float(start_y + dy)))


    # Return points
    return {
        "left": left_line_points,
        "middle": middle_line_points,
        "right": right_line_points,
        "end_x": float(start_x),
        "end_y": float(start_y - length_pixels),
        "end_direction": float(direction),

    }

def curved_road(amplitude, wavelength, start_x, start_y, length, direction):
    """
    Generate a curved road segment using a sinusoidal function that progresses upward.

    Args:
        amplitude (float): Amplitude of the sine wave in meters (controls the sharpness of the curve).
        wavelength (float): Wavelength of the sine wave in meters (controls the length of the curve).
        start_x (float): Starting x-coordinate in pixels.
        start_y (float): Starting y-coordinate in pixels.
        length (float): Length of the curve in meters.
        direction (float): Starting direction in radians.

    Returns:
        dict: Points for the left, middle, and right lanes.
    """
    # Convert length, amplitude, and wavelength to pixels
    length_pixels = length * config.METER_TO_PIXEL
    amplitude_pixels = amplitude * config.METER_TO_PIXEL
    wavelength_pixels = wavelength * config.METER_TO_PIXEL

    # Generate points
    left_line_points = []
    middle_line_points = []
    right_line_points = []

    num_points = int(length_pixels / 10)  # Step size = 10 pixels
    for i in range(num_points + 1):
        # Compute the y-coordinate along the length (upward progression)
        y = start_y - i * (length_pixels / num_points)

        # Using cosine to generate a sinusoidal curve with smooth transitions
        x_offset = amplitude_pixels * (1 - np.cos(2 * np.pi * (start_y - y) / wavelength_pixels)) / 2


        # Adjust the x-coordinates for left, middle, and right lanes
        middle_x = start_x + x_offset
        left_x = middle_x - config.LANE_WIDTH / 2
        right_x = middle_x + config.LANE_WIDTH / 2

        # Append points
        left_line_points.append((left_x, y))
        middle_line_points.append((middle_x, y))
        right_line_points.append((right_x, y))

    # Return final road data
    return {
        "left": left_line_points,
        "middle": middle_line_points,
        "right": right_line_points,
        "end_x": float(middle_line_points[-1][0]),
        "end_y": float(middle_line_points[-1][1]),
        "end_direction": float(direction),  # Direction remains unchanged for sinusoidal curves
    }

def build_road():
    """
    Build a complete road by combining straight and curved segments.

    Args:
        meter_to_pixel (float): Conversion factor from meters to pixels.
        lane_width (float): Width of the lane in pixels.

    Returns:
        dict: Combined points for left, middle, and right lanes.
    """
    road_segments = []

    # Start position and direction
    start_x = config.CAR_POSITION_ROAD_INIT[0] - config.CAR_WIDTH/1.2  # Center of the screen in x
    start_y = config.SCREEN_HEIGHT  # Start at the bottom of the screen
    direction = 0  # Initially pointing straight down (vertical)

    # Add a straight segment (100 meters)
    segment1 = straight_line_road(20, start_x, start_y, direction)
    road_segments.append(segment1)

    print(segment1["end_x"], segment1["end_y"], segment1["end_direction"])

    # Add a curved segment (45° curve with a radius of 50 meters)
    segment2 = curved_road(
        amplitude=-3,
        wavelength=30,
        start_x=segment1["end_x"],
        start_y=segment1["end_y"],
        length=30,
        direction=segment1["end_direction"]
    )
    road_segments.append(segment2)
    print(segment2["end_x"], segment2["end_y"], segment2["end_direction"])


    # Add another straight segment (50 meters)
    segment3 = straight_line_road(
        length=10, start_x=segment2["end_x"], start_y=segment2["end_y"], direction=segment2["end_direction"])
    road_segments.append(segment3)

    # Add a half-turn segment
    segment4 = curved_road_half_turn(amplitude=-3, start_x=segment3["end_x"], start_y=segment3["end_y"], length=10, direction=segment3["end_direction"])
    road_segments.append(segment4)

    # Add another straight segment (50 meters)
    segment5 = straight_line_road(
        length=10, start_x=segment4["end_x"], start_y=segment4["end_y"], direction=segment4["end_direction"])
    road_segments.append(segment5)

    # Add a half-turn segment
    segment6 = curved_road_half_turn(amplitude=-3, start_x=segment5["end_x"], start_y=segment5["end_y"], length=10, direction=segment5["end_direction"])
    road_segments.append(segment6)

    segment7 = straight_line_road(
        length=30, start_x=segment6["end_x"], start_y=segment6["end_y"], direction=segment6["end_direction"])
    road_segments.append(segment7)

    # Combine all points into a single list for rendering
    combined_left = []
    combined_middle = []
    combined_right = []
    # combined_LTA_left = []
    # combined_LTA_right = []

    for segment in road_segments:
        combined_left.extend(segment["left"])
        combined_middle.extend(segment["middle"])
        combined_right.extend(segment["right"])

        # Compute LTA points (offsets from middle and right lanes)
        # combined_LTA_left.extend([(x + config.LTA_DISTANCE, y) for x, y in segment["middle"]])
        # combined_LTA_right.extend([(x - config.LTA_DISTANCE, y) for x, y in segment["right"]])

    return {
        "left": combined_left,
        "middle": combined_middle,
        "right": combined_right,
        #"LTA_left": combined_LTA_left,
        #"LTA_right": combined_LTA_right,
    }


def scroll_road_data(road_data, offset):
    """
    Adjust road data y-coordinates by adding the offset and truncate LTA lines.
    Args:
        road_data (dict): Precomputed road data containing lane points.
        offset (float): Current scrolling offset in pixels.
    Returns:
        dict: Updated road data with adjusted y-coordinates.
    """
    updated_data = {}
    for key, points in road_data.items():
        # Adjust y-coordinates for scrolling
        updated_points = [(x, y + offset) for x, y in points]
        # Truncate LTA lines to be INSIDE the sensor lookahead
        if key in ["LTA_left", "LTA_right"]:
            updated_points = [
                (x, y) for x, y in updated_points if config.CAR_POSITION_SCREEN_INIT[1] - config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL <= y <= config.SCREEN_HEIGHT
            ]
        updated_data[key] = updated_points
    return updated_data

def draw_car(screen, car_position, theta):    
    """
    Draw the car using an image and display only the back points in different colors.
    :param screen: Pygame screen object
    :param car_position: (x, y) position of the car
    :param theta: Orientation of the car (radians)
    :return: List of back points in screen coordinates.
    """

    # Rotate the car image to match the orientation
    rotated_car = pygame.transform.rotate(config.CAR_IMAGE, -math.degrees(theta))

    # Get the new rectangle of the rotated image to center it correctly
    car_rect = rotated_car.get_rect(center=(car_position[0], car_position[1]))

    # Draw the car image on the screen
    screen.blit(rotated_car, car_rect.topleft)

    # Adjusted car dimensions for better fit
    width_factor = 0.65  # Adjust width scaling factor
    length_factor = 0.8  # Adjust length scaling factor
    half_width = (config.CAR_WIDTH / 2) * width_factor
    half_length = (config.CAR_LENGTH / 2) * length_factor

    # Calculate the back corner points (relative to the car's center)
    front_points = [
        (half_length, -half_width),  # Rear-left
        (half_length, half_width)   # Rear-right
    ]

    # Rotate and translate the back points to screen coordinates
    rotated_front_points = []
    for dx, dy in front_points:
        rotated_x = math.cos(theta) * dx - math.sin(theta) * dy + car_position[0]
        rotated_y = math.sin(theta) * dx + math.cos(theta) * dy + car_position[1]
        rotated_front_points.append((rotated_x, rotated_y))

    # Draw the back points with different colors
    # for point in rotated_front_points:
    #     pygame.draw.circle(screen, config.YELLOW, (int(point[0]), int(point[1])), 5)

    # Return the back points in screen coordinates
    return rotated_front_points

def draw_road(screen, road_data):
    """
    Draws the entire road on the screen based on precomputed road data.

    Args:
        screen: Pygame screen object for drawing.
        road_data (dict): Precomputed road data containing lane points (with offset applied).
    """
    # Draw the right lane line (white)
    for i in range(len(road_data["right"]) - 1):
        pygame.draw.line(screen, config.WHITE, road_data["right"][i], road_data["right"][i + 1], 5)

    # Draw the left lane line (white)
    for i in range(len(road_data["left"]) - 1):
        pygame.draw.line(screen, config.WHITE, road_data["left"][i], road_data["left"][i + 1], 5)

    # Draw the middle lane line (yellow, dashed)
    for i in range(0, len(road_data["middle"]) - 1, 2):  # Dashed pattern
        pygame.draw.line(screen, config.YELLOW, road_data["middle"][i], road_data["middle"][i + 1], 3)

    # Draw LTA left and right lines
    for i in range(0, len(road_data["LTA_left"]) - 1, 1):
        pygame.draw.line(screen, config.RED, road_data["LTA_left"][i], road_data["LTA_left"][i + 1], 2)

    for i in range(0, len(road_data["LTA_right"]) - 1, 1):
        pygame.draw.line(screen, config.RED, road_data["LTA_right"][i], road_data["LTA_right"][i + 1], 2)

def calculate_closest_distance(car_x, car_y, lane_points):
    """
    Calculate the closest perpendicular distance from a car point to a list of lane points.
    
    Args:
        car_x (float): x-coordinate of the car point.
        car_y (float): y-coordinate of the car point.
        lane_points (list): List of (x, y) points representing a lane line.
    
    Returns:
        float: Closest distance from the car point to the lane line.
    """
    min_distance = float('inf')
    for lane_x, lane_y in lane_points:
        distance = np.sqrt((car_x - lane_x)**2 + (car_y - lane_y)**2)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def plot_and_save_distances(time_series, noisy_middle, noisy_right, optimal_distance, filename="distance_plot.png"):

    """
    Plot the distances over time and save the figure.
    """
    print("Plotting and saving distances...")
    noisy_middle_scaled = np.array(noisy_middle) * config.PIXEL_TO_METER  
    noisy_right_scaled = np.array(noisy_right) * config.PIXEL_TO_METER
    
    # Plot the scaled distances
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, noisy_middle_scaled, label="Distance to Middle Lane", color="yellow")
    plt.plot(time_series, noisy_right_scaled, label="Distance to Right Lane", color="grey")
    plt.plot(time_series, (np.full(len(time_series), config.LTA_TOLERANCE*config.PIXEL_TO_METER)), label="LTA Threshold", color="red",linestyle="--")
    plt.plot(time_series, (np.full(len(time_series), optimal_distance*config.PIXEL_TO_METER)), label="Optimal distance", color="green",linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distances to Lane Boundaries Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

def curved_road_half_turn(amplitude, start_x, start_y, length, direction):
    """
    Generate a half-turn road segment using a cosine function for smooth transitions.

    Args:
        amplitude (float): Amplitude of the curve in meters (controls the sharpness of the curve).
        start_x (float): Starting x-coordinate in pixels.
        start_y (float): Starting y-coordinate in pixels.
        length (float): Total forward length of the curve in meters.
        direction (float): Starting direction in radians.

    Returns:
        dict: Points for the left, middle, and right lanes.
    """
    # Convert length and amplitude to pixels
    length_pixels = length * config.METER_TO_PIXEL
    amplitude_pixels = amplitude * config.METER_TO_PIXEL

    # Generate points
    left_line_points = []
    middle_line_points = []
    right_line_points = []

    num_points = int(length_pixels / 10)  # Step size = 10 pixels
    for i in range(num_points + 1):
        # Compute the y-coordinate along the length (upward progression)
        y = start_y - i * (length_pixels / num_points)

        # Using half cosine for smooth half-turn transition
        x_offset = amplitude_pixels * (1 - np.cos(np.pi * i / num_points)) / 2

        # Adjust the x-coordinates for left, middle, and right lanes
        middle_x = start_x + x_offset
        left_x = middle_x - config.LANE_WIDTH / 2
        right_x = middle_x + config.LANE_WIDTH / 2

        # Append points
        left_line_points.append((left_x, y))
        middle_line_points.append((middle_x, y))
        right_line_points.append((right_x, y))

    # Return final road data
    return {
        "left": left_line_points,
        "middle": middle_line_points,
        "right": right_line_points,
        "end_x": float(middle_line_points[-1][0]),
        "end_y": float(middle_line_points[-1][1]),
        "end_direction": float(direction),  # Direction remains unchanged for sinusoidal curves
    }

def add_segment(road_segments, segment_type, **kwargs):
    """
    Adds a road segment to the list and updates the end position and direction.

    Args:
        road_segments (list): List of road segments.
        segment_type (function): Function to generate the road segment (e.g., straight_line_road, curved_road).
        **kwargs: Parameters specific to the segment type (length, amplitude, etc.).
    """
    # Use the last segment's end position and direction as the start for the new segment
    if road_segments:
        last_segment = road_segments[-1]
        kwargs["start_x"] = last_segment["end_x"]
        kwargs["start_y"] = last_segment["end_y"]
        kwargs["direction"] = last_segment["end_direction"]
    
    # Generate the new segment and add it to the list
    new_segment = segment_type(**kwargs)
    road_segments.append(new_segment)

def draw_sensor_cone(screen, car_position, theta, fov_angle):
    """
    Draw a semi-transparent yellow cone with concentric arcs representing the car's sensor range.
    """
    car_x, car_y = car_position

    # Use the same lookahead distance as the LTA lines
    lookahead_distance = config.SENSOR_LOOKAHEAD * config.METER_TO_PIXEL

    # Convert fov_angle to radians
    fov_radians = math.radians(fov_angle)

    # Create a semi-transparent surface for the cone
    cone_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    # Draw the main filled cone
    cone_points = [(car_x, car_y)]
    num_points = 50  # Number of points to approximate the arc
    for i in range(num_points + 1):
        angle = theta - fov_radians / 2 + i * (fov_radians / num_points)
        x = car_x + lookahead_distance * math.cos(angle)
        y = car_y + lookahead_distance * math.sin(angle)
        cone_points.append((x, y))
    cone_color = (255, 255, 0, 40)  # Bright yellow with 20% transparency
    pygame.draw.polygon(cone_surface, cone_color, cone_points)

    # Draw concentric arcs within the cone
    num_arcs = 4  # Number of concentric arcs
    arc_color = (255, 255, 0, 90)  # Slightly brighter yellow
    for arc_index in range(1, num_arcs + 1):
        arc_radius = lookahead_distance * (arc_index / num_arcs)
        arc_points = []
        for i in range(num_points + 1):
            angle = theta - fov_radians / 2 + i * (fov_radians / num_points)
            x = car_x + arc_radius * math.cos(angle)
            y = car_y + arc_radius * math.sin(angle)
            arc_points.append((x, y))
        pygame.draw.lines(cone_surface, arc_color, False, arc_points, 2)  # Draw the arc

    # Blit the cone surface onto the screen
    screen.blit(cone_surface, (0, 0))

def generate_random_trees(road_data, tree_images, tree_density=0.2, safe_margin=50):
    """
    Generate random tree positions along the road, avoiding the road area.

    Args:
        road_data (dict): The road data containing left and right lane points.
        tree_images (list): List of preloaded tree sprite images.
        tree_density (float): Fraction of road points that will have trees.
        safe_margin (int): Minimum distance from the road (in pixels).

    Returns:
        list: List of tree data [(image, x, y), ...].
    """
    tree_positions = []

    # Iterate through the road points and randomly decide to place trees
    for i, (left_x, left_y) in enumerate(road_data["left"]):
        if random.random() < tree_density:  # Random chance for a tree on the left
            random_x = random.uniform(0, left_x - safe_margin)  # Left of the road
            chosen_tree = random.choice(tree_images)  # Randomly choose tree image
            tree_positions.append((chosen_tree, random_x, left_y))

    for i, (right_x, right_y) in enumerate(road_data["right"]):
        if random.random() < tree_density:  # Random chance for a tree on the right
            random_x = random.uniform(right_x + safe_margin, config.SCREEN_WIDTH)  # Right of the road
            chosen_tree = random.choice(tree_images)  # Randomly choose tree image
            tree_positions.append((chosen_tree, random_x, right_y))

    return tree_positions

def draw_trees(screen, tree_positions, offset):
    """
    Draw trees on the screen based on their positions.

    Args:
        screen: Pygame screen object.
        tree_positions (list): List of tree data [(image, x, y), ...].
        offset (float): Vertical scrolling offset for the road.
    """
    for tree_image, x, y in tree_positions:
        # Adjust tree position with the scrolling offset
        adjusted_y = y + offset
        if 0 <= adjusted_y <= config.SCREEN_HEIGHT:  # Only draw visible trees
            screen.blit(tree_image, (x, adjusted_y))
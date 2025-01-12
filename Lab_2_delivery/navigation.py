import numpy as np
import pygame
import matplotlib.pyplot as plt
import config
from scipy.interpolate import CubicSpline
import robot
import window


def get_optimal_trajectory(LTA_left_detection_processed, LTA_right_detection_processed):
    """
    Extract reference trajectory (target positin and orientation) from lane points.
    Input
    - LTA_left_points: List of points the right of the yellow line
    - LTA_right_points: Points the left of the right line
    - car_position: Current position of the car
    Output
    - Optimal trajectory for car position
    - Optimal angles for car orientation
    """         
    optimal_trajectory = []
    optimal_theta = []
    
    # Ensure both lists have the same length
    assert len(LTA_left_detection_processed) == len(LTA_right_detection_processed), "LTA boundaries must have the same number of points."
    
    # Calculate the midpoint between LTA boundaries
    for left, right in zip(LTA_left_detection_processed, LTA_right_detection_processed):
        if left[1] < config.CAR_POSITION_SCREEN_INIT[1] or right[1] < config.CAR_POSITION_SCREEN_INIT[1]:
            mid_x = (left[0] + right[0]) / 2
            mid_y = (left[1] + right[1]) / 2
            optimal_trajectory.append((mid_x, mid_y))
    
    # Calculate angles based on the trajectory
    for i in range(len(optimal_trajectory) - 1):
        x1, y1 = optimal_trajectory[i]
        x2, y2 = optimal_trajectory[i + 1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        optimal_theta.append(angle)
    
    # Add a dummy angle (or repeat the last one) for the last point
    if optimal_theta:
        optimal_theta.append(optimal_theta[-1])
    else:
        optimal_theta.append(0.0)  # Default angle if there are no points

    
    return optimal_trajectory, optimal_theta #TODO: not dependent on time



def recovery_trajectory_spline(position, car_y_screen, theta_start, phi_start, dt, T, V, optimal_thetas, optimal_trajectory, lookahead=200):
   
    length_of_trajectory = len(optimal_trajectory)
    middle_index = len(optimal_trajectory) // 2

    x_start = position[0]
    y_start = np.copy(car_y_screen)

    x_mid = optimal_trajectory[middle_index // 2][0]
    y_mid = optimal_trajectory[middle_index // 2][1]

    x_end = optimal_trajectory[middle_index][0]
    y_end = optimal_trajectory[middle_index][1]  

    intermediate_x = optimal_trajectory[middle_index + middle_index//2][0]
    intermediate_y = optimal_trajectory[middle_index + middle_index//2][1]

    # # Combine waypoints
    # waypoints_y = np.concatenate(([y_end], intermediate_y, [y_mid, y_start]))
    # waypoints_x = np.concatenate(([x_end], intermediate_x, [x_mid, x_start]))

    waypoints_y = [y_end, y_mid, intermediate_y, y_start]  # Reversed order for increasing y
    waypoints_x = [x_end, x_mid, intermediate_x, x_start]  # Reversed order for increasing y

    sorted_indices = np.argsort(waypoints_y)

    x_sorted = np.array(waypoints_x)[sorted_indices]
    y_sorted = np.array(waypoints_y)[sorted_indices]

    v_y = robot.get_velocity_y_direction(V, theta_start, phi_start)
    
    # Fit a cubic spline with increasing y values
    # boundary = ((1, np.tan(optimal_thetas[-1])), (1, np.tan(theta_start))) # Curves
    boundary = 'natural' # Straight line
    spline = CubicSpline(y_sorted, x_sorted, bc_type= boundary)

    # # Generate evaluation points with reversed y-direction
    y_eval = np.linspace(y_end, y_start, 100)
    x_eval = spline(y_eval)

    x_opt = [x for x,y in optimal_trajectory]
    y_opt = [y for x,y in optimal_trajectory]

    # Plot the spline
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_eval, y_eval, label="Recovery Spline", color="blue")
    # plt.scatter(waypoints_x, waypoints_y, color="red", label="Waypoints", zorder=5)
    # plt.scatter(x_opt, y_opt, color="green", linestyle="--", label="Optimal trajectory")
    # plt.title("Recovery Spline Trajectory (Rotated)")
    # plt.xlabel("Lateral Position (x)")
    # plt.ylabel("Forward Position (y)")
    # plt.legend()
    # plt.grid()
    # plt.gca().invert_yaxis()
    # plt.show()

    # Generate trajectory points with inverted y-axis
    recovery_trajectory = []
    t = 0.0
    while t <= T:
        y = y_start + v_y * t
        if y < y_end:
            break
        x = spline(y)

        recovery_trajectory.append((float(x), y))
        t += dt


    # Calculate angles based on the trajectory
    recovery_thetas = []
    recovery_thetas.append(theta_start)
    
    for i in range(len(recovery_trajectory) - 1):
        x1, y1 = recovery_trajectory[i]
        x2, y2 = recovery_trajectory[i + 1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        recovery_thetas.append(angle)

    y_eval = [y for x,y in recovery_trajectory]
    x_eval = [x for x,y in recovery_trajectory]


    # # Plot the spline
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_eval, y_eval, label="Recovery Spline", color="blue")
    # plt.scatter(x_eval, y_eval, color="red", label="Waypoints", zorder=5)
    # plt.scatter(x_opt, y_opt, color="green", linestyle="--", label="Optimal trajectory")
    # plt.title("Recovery Spline Trajectory (Rotated)")
    # plt.xlabel("Lateral Position (x)")
    # plt.ylabel("Forward Position (y)")
    # plt.legend()
    # plt.grid()
    # plt.gca().invert_yaxis()
    # plt.show()

    return recovery_trajectory, recovery_thetas

def predict_trajectory(position_road, car_y_screen, theta, phi, dphi, horizon): 
    """
    Assuming dphi will decline as nobody is stearing the wheel.
    """
    car_position_screen = np.copy(position_road)
    y_car_screen = np.copy(car_y_screen)
    car_position_screen[1] = y_car_screen
    theta_local = float(np.copy(theta))
    phi_local = float(np.copy(phi))
    dphi_local = float(np.copy(dphi))

    car_positions = []
    theta_array = []

    # Simulate future variables
    for _ in range(horizon):
        car_position_screen, _, theta_local, phi_local = robot.update_kinematics(config.V, dphi_local, car_position_screen, car_position_screen, theta_local, phi_local)
        car_positions.append(list(np.copy(car_position_screen)))
        theta_array.append(np.copy(theta_local))

    return car_positions, theta_array

def plot_trajectory(trajectory, thetas, color, screen = config.SCREEN):
    """
    Draw the recovery trajectory with orientation arrows.
    """
    if len(trajectory) > 1:
        points = [(int(x), int(y)) for x, y in trajectory]
        pygame.draw.aalines(screen, color, False, points, 2)

        # Draw orientation arrows for better visualization
        for (x, y), theta in zip(trajectory, thetas):
            arrow_x = x + 15 * np.cos(theta)
            arrow_y = y + 15 * np.sin(theta)
            pygame.draw.line(screen, color, (x, y), (arrow_x, arrow_y), 2)


# -----------------------------------
# Function to compute a Bezier curve given control points
def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t_values = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))

    for i, t in enumerate(t_values):
        point = np.zeros(2)
        for j, control_point in enumerate(control_points):
            bernstein_poly = (np.math.comb(n, j) * (t**j) * ((1 - t)**(n - j)))
            point += bernstein_poly * np.array(control_point)
        curve[i] = point

    return curve

# Modify the recovery_trajectory_spline function to use a cubic Bezier curve with dynamic control points
def recovery_trajectory_spline_k(position, car_y_screen, theta_start, phi_start, dt, T, V, optimal_thetas, optimal_trajectory, lookahead=200, left_boundary=False):
    x_start = position[0]
    y_start = np.copy(car_y_screen)

    y_end = y_start - lookahead
    x_end = optimal_trajectory[-1][0]

    # Identify the goal point and orientation on the optimal trajectory
    goal_idx = len(optimal_trajectory) - 1
    goal_x, goal_y = optimal_trajectory[goal_idx]
    goal_theta = optimal_thetas[goal_idx]

    # Control points for cubic Bezier curve (two control points for smoother turns)
    flip_factor = -1 if left_boundary else 1  # Adjust curve direction based on the boundary
    control_x1 = x_start + 30 * np.cos(theta_start)
    control_y1 = y_start + 30 * np.sin(theta_start)
    control_x2 = (x_start + goal_x) / 2 + flip_factor * 50 * np.sin(theta_start)
    control_y2 = (y_start + goal_y) / 2 - flip_factor * 50 * np.cos(theta_start)

    control_points = [
        (x_start, y_start),
        (control_x1, control_y1),
        (control_x2, control_y2),
        (goal_x, goal_y)
    ]

    # Generate the Bezier curve
    curve = bezier_curve(control_points)

    # Plot the Bezier curve along with the optimal trajectory
    # x_opt = [x for x, y in optimal_trajectory]
    # y_opt = [y for x, y in optimal_trajectory]

    # plt.figure(figsize=(8, 6))
    # plt.plot(curve[:, 0], curve[:, 1], label="Bezier Recovery Trajectory", color="blue")
    # plt.scatter(*zip(*control_points), color="red", label="Control Points", zorder=5)
    # plt.plot(x_opt, y_opt, color="green", linestyle="--", label="Optimal Trajectory")
    # plt.title("Recovery Trajectory using Bezier Curve")
    # plt.xlabel("Lateral Position (x)")
    # plt.ylabel("Forward Position (y)")
    # plt.legend()
    # plt.grid()
    # plt.gca().invert_yaxis()
    # plt.show()

    # Generate trajectory points
    recovery_trajectory = [(float(x), float(y)) for x, y in curve]

    # Calculate angles based on the trajectory
    recovery_thetas = [theta_start]
    for i in range(len(recovery_trajectory) - 1):
        x1, y1 = recovery_trajectory[i]
        x2, y2 = recovery_trajectory[i + 1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        recovery_thetas.append(angle)

    return recovery_trajectory, recovery_thetas


# -----------------------------------

'''
def bezier_curve(p0, p1, p2, t):
    """ Quadratic Bézier curve equation """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
'''


# def get_recovery_trajectory(car_position_road, car_y_screen, car_theta, 
#                             optimal_trajectory, optimal_thetas, 
#                             left_boundary=False):
#     """
#     Generate a smooth recovery trajectory using a cubic Bézier curve.
#     Returns both recovery positions and orientations.
#     Flips the trajectory horizontally if is_flipped=True.
#     """
#     car_x = car_position_road[0] # TODO: RYDD
#     car_y = car_y_screen
    
#     steps = len(optimal_trajectory)
    
#     # Identify the goal point and orientation on the optimal trajectory
#     goal_idx = len(optimal_trajectory) - 1
#     goal_x, goal_y = optimal_trajectory[goal_idx]
#     goal_theta = optimal_thetas[goal_idx]

#     # Control points for cubic Bézier curve (two control points for smoother turns)
#     flip_factor = -1 if left_boundary else 1  # Adjust curve direction for flipping, depending on weither the car is driving towards/over the left or right LTA-detection line
#     control_x1 = car_x + 30 * np.cos(car_theta)
#     control_y1 = car_y + 30 * np.sin(car_theta)
#     control_x2 = (car_x + goal_x) / 2 + flip_factor * 50 * np.sin(car_theta)
#     control_y2 = (car_y + goal_y) / 2 - flip_factor * 50 * np.cos(car_theta)

#     recovery_positions = []
#     recovery_thetas = []

#     # Compute the initial desired angle for turning correction
#     angle_to_goal = np.arctan2(goal_y - car_y, goal_x - car_x)
#     theta = car_theta  # Initialize theta correctly

#     # Generate the smooth cubic Bézier curve and theta adjustment
#     for t in np.linspace(0, 1, steps):
#         # Cubic Bézier interpolation
#         x = ((1 - t)**3 * car_x + 
#              3 * (1 - t)**2 * t * control_x1 +
#              3 * (1 - t) * t**2 * control_x2 +
#              t**3 * goal_x)

#         y = ((1 - t)**3 * car_y + 
#              3 * (1 - t)**2 * t * control_y1 +
#              3 * (1 - t) * t**2 * control_y2 +
#              t**3 * goal_y)

#         # Adjust theta progressively
#         if t < 0.9:
#             error_theta = angle_to_goal - theta
#             theta += 0.2 * error_theta
#         else:
#             error_theta = goal_theta - theta
#             theta += 0.5 * error_theta  # More decisive correction near the goal

#         recovery_positions.append((x, y))
#         recovery_thetas.append(theta)

#     return recovery_positions, recovery_thetas #, (goal_x, goal_y)

# def get_recovery_trajectory_gammel(car_x, car_y, car_theta, optimal_trajectory, recovery_goal_offset=20, curvature_factor=50):
#     """
#     Calculate a recovery trajectory that smoothly brings the car back to the optimal trajectory.
    
#     Args:
#         car_x (float): Current x-coordinate of the car.
#         car_y (float): Current y-coordinate of the car.
#         optimal_trajectory (list): List of (x, y) points representing the optimal trajectory.
#         recovery_goal_offset (int): Distance ahead on the optimal trajectory to set as the goal.
#         damping_factor (float): Controls the smoothness of the recovery trajectory (0.0 to 1.0).

#     Returns:
#         list: Recovery trajectory as a list of (x, y) points.
#     """
#     # Find the closest point on the optimal trajectory
#     distances = [np.sqrt((car_x - x)**2 + (car_y - y)**2) for x, y in optimal_trajectory]
#     closest_idx = np.argmin(distances)

#     # Set the recovery goal a bit ahead of the car's current position
#     recovery_goal_idx = min(closest_idx + recovery_goal_offset, len(optimal_trajectory) - 1)
#     recovery_goal = optimal_trajectory[recovery_goal_idx]

#     # Normalize the curve to start at the car's position and end at the recovery goal
#     recovery_positions = []
#     steps = 50  # Number of points in the recovery path

#     for t in np.linspace(0, 1, steps):
#         # Normalized S-curve: y = 3 * (1 - t) * t^2
#         s_curve_y = damping_factor * 3 * (1 - t) * (t ** 2)

#         # Scale and translate the S-curve to fit the car's current position and the recovery goal
#         x = car_x + t * (recovery_goal[0] - car_x) * scale_x
#         y = car_y + s_curve_y * (recovery_goal[1] - car_y) * scale_y

#         recovery_positions.append((x, y))

#     # Calculate orientations (theta) for each segment of the recovery path
#     recovery_angles = []
#     for i in range(len(recovery_positions) - 1):
#         x1, y1 = recovery_positions[i]
#         x2, y2 = recovery_positions[i + 1]
#         theta = np.arctan2(y2 - y1, x2 - x1)  # Angle of the segment
#         recovery_angles.append(theta)

#     # Repeat the last angle for the final point to ensure both lists have the same length
#     if recovery_angles:
#         recovery_angles.append(recovery_angles[-1])
#     else:
#         recovery_angles.append(0.0)  # Default angle if no points

#     return recovery_positions, recovery_angles

# def get_recovery_trajectory_gammel(car_x, car_y, car_theta, optimal_trajectory, recovery_goal_offset=20, curvature_factor=50):
#     """
#     Calculate a recovery trajectory that smoothly brings the car back to the optimal trajectory.
    
#     Args:
#         car_x (float): Current x-coordinate of the car.
#         car_y (float): Current y-coordinate of the car.
#         optimal_trajectory (list): List of (x, y) points representing the optimal trajectory.
#         recovery_goal_offset (int): Distance ahead on the optimal trajectory to set as the goal.
#         damping_factor (float): Controls the smoothness of the recovery trajectory (0.0 to 1.0).

#     Returns:
#         list: Recovery trajectory as a list of (x, y) points.
#     """
#     # Find the closest point on the optimal trajectory
#     distances = [np.sqrt((car_x - x)**2 + (car_y - y)**2) for x, y in optimal_trajectory]
#     closest_idx = np.argmin(distances)

#     # Set the recovery goal a bit ahead of the car's current position
#     recovery_goal_idx = min(closest_idx + recovery_goal_offset, len(optimal_trajectory) - 1)
#     recovery_goal = optimal_trajectory[recovery_goal_idx]

#     # Normalize the curve to start at the car's position and end at the recovery goal
#     recovery_positions = []
#     steps = config.SENSOR_LOOKAHEAD  # Number of points in the recovery path
    
#     # Calculate the angle between the car's current orientation (theta) and the recovery goal
#     goal_angle = np.arctan2(recovery_goal[1] - car_y, recovery_goal[0] - car_x)
#     angle_diff = goal_angle - car_theta

#     # Smooth the angle difference to avoid sharp turns
#     angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

#     # Create control points for a curved path from the car's current position to the goal
#     curve_control_x = car_x + curvature_factor * np.cos(car_theta + np.pi / 2) * np.abs(angle_diff)
#     curve_control_y = car_y + curvature_factor * np.sin(car_theta + np.pi / 2) * np.abs(angle_diff)

#     # Generate the recovery trajectory using a smooth curve
#     recovery_positions = []
#     for t in np.linspace(0, 1, steps):
#         # Cubic Bézier interpolation (smooth transition)
#         x = (1 - t)**2 * car_x + 2 * (1 - t) * t * curve_control_x + t**2 * recovery_goal[0]
#         y = (1 - t)**2 * car_y + 2 * (1 - t) * t * curve_control_y + t**2 * recovery_goal[1]
#         recovery_positions.append((x, y))
        
        
#     # Calculate orientations (theta) for each segment of the recovery path
#     recovery_angles = []
#     for i in range(len(recovery_positions) - 1):
#         x1, y1 = recovery_positions[i]
#         x2, y2 = recovery_positions[i + 1]
#         theta = np.arctan2(y2 - y1, x2 - x1)  # Angle of the segment
#         recovery_angles.append(theta)

#     # Repeat the last angle for the final point to ensure both lists have the same length
#     if recovery_angles:
#         recovery_angles.append(recovery_angles[-1])
#     else:
#         recovery_angles.append(0.0)  # Default angle if no points

#     return recovery_positions, recovery_angles, recovery_goal
    
# def get_recovery_trajectory_hmmm(car_x, car_y, car_theta, 
#                             optimal_trajectory, optimal_thetas, 
#                             recovery_goal_offset=20, steps=50):
#     """
#     Generate a smooth recovery trajectory using a quadratic Bézier curve.
#     Returns both recovery positions and orientations.
#     """
#     # Identify the goal point and orientation on the optimal trajectory
#     distances = [np.hypot(car_x - x, car_y - y) for x, y in optimal_trajectory]
#     closest_idx = np.argmin(distances)
#     goal_idx = min(closest_idx + recovery_goal_offset, len(optimal_trajectory) - 1)
#     goal_x, goal_y = optimal_trajectory[goal_idx]
#     goal_theta = optimal_thetas[goal_idx]

#     # Control point for smoother Bézier curve
#     control_x = (car_x + goal_x) / 2 + 30 * np.sin(car_theta)
#     control_y = (car_y + goal_y) / 2 - 30 * np.cos(car_theta)

#     recovery_positions = []
#     recovery_thetas = []

#     # Compute the initial desired angle for turning correction
#     angle_to_goal = np.arctan2(goal_y - car_y, goal_x - car_x)
#     theta = car_theta  # Initialize theta correctly

#     # Generate the smooth Bézier curve and theta adjustment
#     for t in np.linspace(0, 1, steps):
#         if t < steps*0.6:
#             # Quadratic Bézier interpolation
#             x = (1 - t)**2 * car_x + 2 * (1 - t) * t * control_x + t**2 * goal_x
#             y = (1 - t)**2 * car_y + 2 * (1 - t) * t * control_y + t**2 * goal_y

#             # Adjust theta more logically: point towards the goal
#             error_theta = angle_to_goal - theta
#             theta += 0.2 * error_theta  # More decisive correction towards the goal

#             recovery_positions.append((x, y))
#             recovery_thetas.append(theta)
#         else:
#             # Quadratic Bézier interpolation
#             x = (1 - t)**2 * car_x + 2 * (1 - t) * t * control_x + t**2 * goal_x
#             y = (1 - t)**2 * car_y + 2 * (1 - t) * t * control_y + t**2 * goal_y

#             # Gradually adjust theta towards goal_theta more progressively
#             error_theta = goal_theta - theta
#             theta += error_theta  # Increasing weight towards the end

#             recovery_positions.append((x, y))
#             recovery_thetas.append(theta)

#     return recovery_positions, recovery_thetas, (goal_x, goal_y)

# def get_recovery_trajectory_good(car_x, car_y, car_theta, 
#                             optimal_trajectory, optimal_thetas, steps=50):
#     """
#     Generate a smooth recovery trajectory using a cubic Bézier curve.
#     Returns both recovery positions and orientations.
#     """
#     # Identify the goal point and orientation on the optimal trajectory
#     goal_idx = len(optimal_trajectory) - 1
#     goal_x, goal_y = optimal_trajectory[goal_idx]
#     goal_theta = optimal_thetas[goal_idx]

#     # Control points for cubic Bézier curve (two control points for smoother turns)
#     control_x1 = car_x + 30 * np.cos(car_theta)
#     control_y1 = car_y + 30 * np.sin(car_theta)
#     control_x2 = (car_x + goal_x) / 2 + 50 * np.sin(car_theta)
#     control_y2 = (car_y + goal_y) / 2 - 50 * np.cos(car_theta)

#     recovery_positions = []
#     recovery_thetas = []

#     # Compute the initial desired angle for turning correction
#     angle_to_goal = np.arctan2(goal_y - car_y, goal_x - car_x)
#     theta = car_theta  # Initialize theta correctly

#     # Generate the smooth cubic Bézier curve and theta adjustment
#     for t in np.linspace(0, 1, steps):
#         # Cubic Bézier interpolation
#         x = ((1 - t)**3 * car_x + 
#              3 * (1 - t)**2 * t * control_x1 +
#              3 * (1 - t) * t**2 * control_x2 +
#              t**3 * goal_x)

#         y = ((1 - t)**3 * car_y + 
#              3 * (1 - t)**2 * t * control_y1 +
#              3 * (1 - t) * t**2 * control_y2 +
#              t**3 * goal_y)

#         # Adjust theta progressively
#         if t < 0.9:
#             error_theta = angle_to_goal - theta
#             theta += 0.2 * error_theta
#         else:
#             error_theta = goal_theta - theta
#             theta += 0.5 * error_theta  # More decisive correction near the goal

#         recovery_positions.append((x, y))
#         recovery_thetas.append(theta)

#     return recovery_positions, recovery_thetas, (goal_x, goal_y)

# def get_recovery_trajectory(car_x, car_y, car_theta, 
#                             optimal_trajectory, optimal_thetas, 
#                             left_boundary=False):
#     """
#     Generate a smooth recovery trajectory using a cubic Bézier curve.
#     Returns both recovery positions and orientations.
#     Flips the trajectory horizontally if `is_flipped=True`.
#     """
    
#     steps = len(optimal_trajectory)
    
#     # Identify the goal point and orientation on the optimal trajectory
#     goal_idx = len(optimal_trajectory) - 1
#     goal_x, goal_y = optimal_trajectory[goal_idx]
#     goal_theta = optimal_thetas[goal_idx]

#     # Control points for cubic Bézier curve (two control points for smoother turns)
#     flip_factor = -1 if left_boundary else 1  # Adjust curve direction for flipping, depending on weither the car is driving towards/over the left or right LTA-detection line
#     control_x1 = car_x + 30 * np.cos(car_theta)
#     control_y1 = car_y + 30 * np.sin(car_theta)
#     control_x2 = (car_x + goal_x) / 2 + flip_factor * 50 * np.sin(car_theta)
#     control_y2 = (car_y + goal_y) / 2 - flip_factor * 50 * np.cos(car_theta)

#     recovery_positions = []
#     recovery_thetas = []

#     # Compute the initial desired angle for turning correction
#     angle_to_goal = np.arctan2(goal_y - car_y, goal_x - car_x)
#     theta = car_theta  # Initialize theta correctly

#     # Generate the smooth cubic Bézier curve and theta adjustment
#     for t in np.linspace(0, 1, steps):
#         # Cubic Bézier interpolation
#         x = ((1 - t)**3 * car_x + 
#              3 * (1 - t)**2 * t * control_x1 +
#              3 * (1 - t) * t**2 * control_x2 +
#              t**3 * goal_x)

#         y = ((1 - t)**3 * car_y + 
#              3 * (1 - t)**2 * t * control_y1 +
#              3 * (1 - t) * t**2 * control_y2 +
#              t**3 * goal_y)

#         # Adjust theta progressively
#         if t < 0.9:
#             error_theta = angle_to_goal - theta
#             theta += 0.2 * error_theta
#         else:
#             error_theta = goal_theta - theta
#             theta += 0.5 * error_theta  # More decisive correction near the goal

#         recovery_positions.append((x, y))
#         recovery_thetas.append(theta)

#     return recovery_positions, recovery_thetas, (goal_x, goal_y)
import matplotlib.pyplot as plt
import numpy as np
import config
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_ref_against_pred_position(x_ref, x_pred, t):
    """
    Plots positions (x) over time (t).

    Parameters:
    x (list or array-like): Positions at different time steps.
    t (list or array-like): Corresponding time values.

    Returns:
    None
    """
    if len(x_ref) != len(t):
        print(f'Length x_ref: {len(x_ref)}, length t: {len(t)} ')
        raise ValueError("x and t must have the same length.")
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_ref, t, label='Recovery trajectory' ,marker='o', color='red')
    plt.plot(x_pred, t, label='Predicted trajectory', marker='o', color='blue')
    plt.plot(x_ref, t, label='Recovery trajectory' ,marker='o', color='red')
    plt.plot(x_pred, t, label='Predicted trajectory', marker='o', color='blue')

    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    plt.title('Recovery vs Predicted Trajectories')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_mpc_trajectories(time_past, time_future, measured_output, predicted_output, 
                          reference_trajectory, past_control_input, predicted_control_input):
    """
    Plots an MPC-style graph with past, present, and future trajectories.

    Parameters:
    - time_past: List or array of past time values.
    - time_future: List or array of future time values.
    - measured_output: List or array of measured output values (past).
    - predicted_output: List or array of predicted output values (future).
    - reference_trajectory: List or array of reference trajectory values (future).
    - past_control_input: List or array of past control input values.
    - predicted_control_input: List or array of predicted control input values.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot measured output (past)
    plt.plot(time_past, measured_output, label="Measured Output", color="orange", marker="o", linestyle="-")
    
    # Plot predicted output (future)
    plt.plot(time_future, predicted_output, label="Predicted Output", color="yellow", marker="o", linestyle="-")
    
    # Plot reference trajectory (future)
    plt.plot(time_future, reference_trajectory, label="Reference Trajectory", color="red", marker="o", linestyle="-")
    
    # Plot past control input
    plt.step(time_past, past_control_input, label="Past Control Input", color="blue", linestyle="-")
    
    # Plot predicted control input
    plt.step(time_future, predicted_control_input, label="Predicted Control Input", color="cyan", linestyle="-")
    
    # Add vertical line to separate past and future
    plt.axvline(x=0, color="black", linestyle="--", label="Current Time")
    
    # Annotations
    plt.text(-1, max(measured_output)*1.1, "PAST", fontsize=12, verticalalignment='center')
    plt.text(1, max(measured_output)*1.1, "FUTURE", fontsize=12, verticalalignment='center')
    
    # Labels and legend
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("MPC Trajectories")
    plt.grid(True)
    plt.legend()
    
    plt.show()

# Example Data
# time_past = np.array([-3, -2, -1, 0])  # Past time steps
# time_future = np.array([1, 2, 3, 4])  # Future time steps

# measured_output = np.array([0, 1, 1.5, 2])  # Measured values
# predicted_output = np.array([2, 2.5, 2.8, 3])  # Predicted future values
# reference_trajectory = np.array([2.2, 2.6, 3, 3.2])  # Reference trajectory

# past_control_input = np.array([0, 1, 1, 2])  # Control inputs in the past
# predicted_control_input = np.array([2, 2, 3, 3])  # Control inputs in the future

# # Call the function
# plot_mpc_trajectories(time_past, time_future, measured_output, predicted_output, 
#                       reference_trajectory, past_control_input, predicted_control_input)



def plot_human_vs_controller_input(time_past, time_future, human_input_past, human_input_future, 
                                   controller_input_past, controller_input_future):
    """
    Plots the control input contributions from the human (dphi) and the controller (dphi_controller) 
    over time, separating past and future.

    Parameters:
    - time_past: List or array of past time values.
    - time_future: List or array of future time values.
    - human_input_past: List or array of past human control input values (dphi).
    - human_input_future: List or array of future human control input values (dphi).
    - controller_input_past: List or array of past controller input values (dphi_controller).
    - controller_input_future: List or array of future controller input values (dphi_controller).
    """
    plt.figure(figsize=(10, 6))
    
    # Plot human input (past and future)
    plt.step(time_past, human_input_past, label="Human Input (Past)", color="orange", linestyle="-")
    plt.step(time_future, human_input_future, label="Human Input (Future)", color="gold", linestyle="--")
    
    # Plot controller input (past and future)
    plt.step(time_past, controller_input_past, label="Controller Input (Past)", color="blue", linestyle="-")
    plt.step(time_future, controller_input_future, label="Controller Input (Future)", color="cyan", linestyle="--")
    
    # Add vertical line to separate past and future
    plt.axvline(x=0, color="black", linestyle="--", label="Current Time")
    
    # Annotations
    plt.text(-1, max(human_input_past + controller_input_past) * 1.1, "PAST", fontsize=12, verticalalignment='center')
    plt.text(1, max(human_input_future + controller_input_future) * 1.1, "FUTURE", fontsize=12, verticalalignment='center')
    
    # Labels and legend
    plt.xlabel("Time")
    plt.ylabel("Control Input")
    plt.title("Human Input vs Controller Input Over Time")
    plt.grid(True)
    plt.legend()
    
    plt.show()

'''
# Example Data
time_past = np.array([-3, -2, -1, 0])  # Past time steps
time_future = np.array([1, 2, 3, 4])  # Future time steps

human_input_past = np.array([0, -1, 1, 2])  # Human inputs (dphi) in the past
human_input_future = np.array([1, 1.5, 2, 2.5])  # Human inputs (dphi) in the future

controller_input_past = np.array([0, 0.5, 1, 1.5])  # Controller inputs (dphi_controller) in the past
controller_input_future = np.array([1.5, 1.8, 2, 2.2])  # Controller inputs (dphi_controller) in the future

# Call the function
plot_human_vs_controller_input(time_past, time_future, human_input_past, human_input_future, 
                               controller_input_past, controller_input_future)
'''


def plot_and_save_distances_with_steering(
    time_series, noisy_middle, noisy_right, optimal_distance,
    human_steering_inputs, controller_steering_inputs, filename="distance_steering_plot.png"
):
    """
    Plot the distances over time and save the figure, including a subplot for steering inputs.
    
    Args:
        time_series (list): Time values for the plot.
        noisy_middle (list): Noisy distances to the middle lane.
        noisy_right (list): Noisy distances to the right lane.
        optimal_distance (float): Optimal distance for the car from the lanes.
        human_steering_inputs (list): Steering inputs from the human driver.
        controller_steering_inputs (list): Steering inputs from the LTA/MPC controller.
        filename (str): File name to save the plot.
    """
    print("Plotting and saving distances with steering inputs...")
    
    noisy_middle_scaled = np.array(noisy_middle) * config.PIXEL_TO_METER  
    noisy_right_scaled = np.array(noisy_right) * config.PIXEL_TO_METER
    
    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    
    # First subplot: Distances to lane boundaries
    axs[0].plot(time_series, noisy_middle_scaled, label="Distance to Middle Line", color="yellow")
    axs[0].plot(time_series, noisy_right_scaled, label="Distance to Right Line", color="grey")
    axs[0].plot(time_series, np.full(len(time_series), config.LTA_TOLERANCE * config.PIXEL_TO_METER), 
                label="LTA Threshold", color="red", linestyle="--")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Distance (m)")
    axs[0].set_title("Distances to Lane Boundaries Over Time")
    axs[0].legend()
    axs[0].grid()
    
    # Second subplot: Steering inputs
    axs[1].plot(time_series, human_steering_inputs, label="Human Steering Input", color="orange")
    axs[1].plot(time_series, controller_steering_inputs, label="Controller Steering Input", color="blue")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Steering Input (dphi)")
    axs[1].set_title("Steering Inputs Over Time")
    axs[1].legend()
    axs[1].grid()
    
    # Save and display the plot
    plt.savefig(filename)
    plt.show()



def render_steering_plot(time_series, human_steering_inputs, controller_steering_inputs, window_size=5):
    """
    Create a smaller plot of steering inputs and render it as a Pygame surface.
    
    Args:
        time_series (list): Time values for the plot.
        human_steering_inputs (list): Steering inputs from the human driver.
        controller_steering_inputs (list): Steering inputs from the LTA/MPC controller.
        window_size (int): Number of seconds to display in the plot.
    
    Returns:
        pygame.Surface: Pygame surface containing the rendered plot.
    """
    # Filter the data to only include the last `window_size` seconds
    if time_series:
        recent_time_index = [i for i, t in enumerate(time_series) if t >= time_series[-1] - window_size]
        time_series = [time_series[i] for i in recent_time_index]
        human_steering_inputs = [human_steering_inputs[i] for i in recent_time_index]
        controller_steering_inputs = [controller_steering_inputs[i] for i in recent_time_index]

    # Create the figure (smaller size)
    fig, ax = plt.subplots(figsize=(2, 1.5))  # Smaller figure size for corner display
    
    # Plot steering inputs
    ax.plot(time_series, human_steering_inputs, label="Human", color="orange", linewidth=1)
    ax.plot(time_series, controller_steering_inputs, label="LTA", color="blue", linewidth=1)
    ax.set_xlabel("Time (s)", fontsize=6)
    ax.set_ylabel("dphi", fontsize=6)
    ax.set_title("Steering Inputs", fontsize=8)
    ax.legend(fontsize=5)
    ax.grid(linewidth=0.4)
    
    # Adjust tick parameters for readability
    ax.tick_params(axis="both", which="major", labelsize=5)

    # Render the plot to a pygame surface
    canvas = FigureCanvas(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    surface = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
    
    plt.close(fig)  # Close the figure to avoid memory leaks
    return surface


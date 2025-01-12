import config

import control
import guidance
import robot 
import sensor
import window 
import documentation as doc

import pygame
import sys
import numpy as np
import math
import time
import sys
import random

# --------------------Initialize pygame---------------------------------
pygame.init()
config.FONT = pygame.font.Font(None, 36)  # Default font, size 36
pygame.display.set_caption("Car Simulation LTA system")
clock = pygame.time.Clock() # Clock for controlling frame rate

# --------------- Initialize variables for car position ----------------
theta, phi, dphi, car_position_road, car_position_screen = robot.init_robot()
window_scrolled = 0 #TODO: variabelnavn?


# Initialize distance recording lists
distance_to_middle = []
distance_to_right = []
time_series = []
human_dphi_values = []
lta_dphi_values = []

# Global recovery flag
recovering_flag = False

# --------------- Generate road with random patches in roadline gone ---------------------
road_with_holes, road_with_holes_indexes = window.generate_road_with_holes_in_roadlines()

# Simulation parameters
start_time = time.time()  # Record the start time
running = True
last_plot_time = 0
print("Starting the simulation...")
#-------------- Main game loop----------------
# Continously run while-loop to make track, gather wheel-input,
#  see whether LTA should be active, and do corrections if needed. And plot.
while running:
    elapsed_time = time.time() - start_time

    if elapsed_time > config.SIMULATION_TIME:
        running = False  # End the simulation

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False # End the simulation

    # Get keys pressed
    keys = pygame.key.get_pressed()

    # ----------- Update and draw road lanes ----------------------------
    window_scrolled += config.V*0.1  # Adjust this value to control the scrolling speed (road-lanes moving vertically)
   
    # Build the road
    scrolled_road = window.scroll_road_data(road_with_holes, window_scrolled)
    config.SCREEN.fill(config.BLACK)

    window.draw_road_holes(config.SCREEN, scrolled_road, road_with_holes_indexes["left_indexes"], road_with_holes_indexes["middle_indexes"], road_with_holes_indexes["right_indexes"])

    # Draw the car
    front_left, front_right = window.draw_car(config.SCREEN, car_position_screen, theta) 
    
    # SENSOR
    # ----------- LTA sensor and processing -----------------------------
    
    LTA_left_detection_noisy, LTA_right_detection_noisy = sensor.LTA_sensor(scrolled_road["middle"], scrolled_road["right"])
    LTA_left_detection_processed, LTA_right_detection_processed, LTA_left_holes_indexes, LTA_right_holes_indexes = sensor.LTA_processing(car_position_screen, LTA_left_detection_noisy, LTA_right_detection_noisy)
    sensor.draw_LTA_detections(config.SCREEN, LTA_left_detection_processed, LTA_right_detection_processed, LTA_left_holes_indexes, LTA_right_holes_indexes)
    
    # LTA threshold
    LTA_left_threshold, LTA_right_threshold = sensor.generate_LTA_threshold(LTA_left_detection_processed, LTA_right_detection_processed)
    sensor.draw_LTA_threshold(config.SCREEN, LTA_left_threshold, LTA_right_threshold)

    
    # ------- Get status of the car --------------------------------
    status = sensor.get_car_bound_status(front_left, front_right, LTA_left_threshold, LTA_right_threshold)

    if status in ["crossing_LTA_threshold_left", "crossing_LTA_threshold_right", "near_LTA_threshold_left", "near_LTA_threshold_right"]: # Trigger recovery if the car is near or outside the road
        recovering_flag = True
    else:
        recovering_flag = False

    # CONTROLLER INPUT
    phi, dphi, human_dphi = control.controller(keys, phi, recovering_flag, status)    

    # GUIDANCE INPUT
    lta_dphi = 0
    if recovering_flag:
        lta_dphi = guidance.get_guidance_input(car_position_road, car_position_screen, theta, phi, dphi, LTA_left_detection_processed, LTA_right_detection_processed, status)
        dphi += lta_dphi

    # ROBOT
    car_position_road, car_position_screen, theta, phi = robot.update_kinematics(config.V, dphi, car_position_road, car_position_screen, theta, phi)

    # SAVE DATA
    human_dphi_values.append(human_dphi)
    lta_dphi_values.append(lta_dphi)

    dist_middle = window.calculate_closest_distance(front_left[0], front_left[1], LTA_left_detection_processed)
    dist_right = window.calculate_closest_distance(front_right[0], front_right[1], LTA_right_detection_processed)


    distance_to_middle.append(dist_middle)
    distance_to_right.append(dist_right)
    time_series.append(elapsed_time)

    # TEXT IN WINDOW
    # text_optimal_steering = config.FONT.render(f"Phi from MPC after max and min: {lta_dphi:.2f}", True, config.WHITE)  # Top-left corner 
    # text_status = config.FONT.render(f"Boundary status: {status}", True, config.WHITE)
    # text_phi = config.FONT.render(f"phi: {phi:.2f}", True, config.WHITE)
    # text_dphi = config.FONT.render(f"dphi: {dphi:.2f}", True, config.WHITE)
    # config.SCREEN.blit(text_phi, (10, 10))           # Top-left corner
    # config.SCREEN.blit(text_dphi, (10, 50))          # Below x 
    # config.SCREEN.blit(text_status, (10, 10))   
    # config.SCREEN.blit(text_optimal_steering, (10, 150))
    
    # -------------------------------------------------------------------

    window.draw_sensor_cone(config.SCREEN, car_position_screen, theta, fov_angle=120)


    # --------------- Comment out the following if the simulation is lagging ----------------------
    # Generate the steering plot for the last 5 seconds
    steering_plot_surface = doc.render_steering_plot(time_series, human_dphi_values, lta_dphi_values, window_size=4)
    # Display the plot in the top-right corner
    plot_x = config.SCREEN.get_width() - steering_plot_surface.get_width() - 10  # 10px padding from the right
    plot_y = 10  # 10px padding from the top
    config.SCREEN.blit(steering_plot_surface, (plot_x, plot_y))
    # ----------------- Light for LTA detection ------------------------t
    base_x = plot_x # + steering_plot_surface.get_width() // 2
    base_y = plot_y + steering_plot_surface.get_height() + 40
    spacing = 60  # Vertical spacing between the lights

    # Draw the three lights with labels "CAR RELATIVE LTA THRESHOLD"
    text1 = config.FONT.render("LTA THRESHOLD STATUS", True, config.WHITE)
    config.SCREEN.blit(text1, (base_x + 70, base_y - 10))

    text2 = config.FONT.render("CROSSING LINE", True, config.WHITE)
    config.SCREEN.blit(text2, (base_x + 120, base_y + spacing - 10))

    text3 = config.FONT.render("NEAR LINE", True, config.WHITE)
    config.SCREEN.blit(text3, (base_x + 120, base_y + 2 * spacing - 10))

    text4 = config.FONT.render("WITHIN LIMITS", True, config.WHITE)
    config.SCREEN.blit(text4, (base_x + 120, base_y + 3 * spacing - 10))

    if status in ["crossing_LTA_threshold_left", "crossing_LTA_threshold_right"]:
        pygame.draw.circle(config.SCREEN, config.WHITE, (base_x + 90, base_y + spacing), 15)  # PAST LINE
    elif status in ["near_LTA_threshold_left", "near_LTA_threshold_right"]:
        pygame.draw.circle(config.SCREEN, config.WHITE, (base_x + 90, base_y + 2* spacing), 15)  # APPROACHING LINE
    else:
        pygame.draw.circle(config.SCREEN, config.WHITE, (base_x + 90, base_y + 3 * spacing), 15)  # INSIDE LINE

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30) # 30

#-------------- End of main game loop----------------
# Quit pygame
pygame.quit()

print("Simulation ended.")

# Add noise to recorded distances
print("Distances recorded.")

# Plot and save the distances


optimal_distance = (distance_to_middle[0]+distance_to_right[0])*config.PIXEL_TO_METER/2
#window.plot_and_save_distances(time_series, noisy_middle, noisy_right, optimal_distance)
doc.plot_and_save_distances_with_steering(
    time_series, distance_to_middle, distance_to_right, config.OPTIMAL_DISTANCE,
    human_dphi_values, lta_dphi_values
)
print("All done.")

sys.exit()






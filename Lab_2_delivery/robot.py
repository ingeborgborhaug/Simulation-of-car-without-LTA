import pygame
import math
import numpy as np
import config



def init_robot():
    theta = config.THETA_INIT
    phi = config.PHI_INIT
    dphi = config.DPHI_INIT
    car_position_road = config.CAR_POSITION_ROAD_INIT #TODO: OBS should car_y_road meybe start at 0 and not 550 like the car_y_screen?
    car_position_screen = config.CAR_POSITION_SCREEN_INIT

    return theta, phi, dphi, car_position_road, car_position_screen

def update_kinematics(V, dphi, car_position_road, car_position_screen, theta, phi):
    """Update the car's position car_position[0]and orientation based on kinematic equations."""
    #global car_position, theta, phi
    
    dt = config.dt    
    phi += dphi *dt
    # Limit steering angle
    phi = max(-config.MAX_PHI, min(config.MAX_PHI, phi))
    
    # Update theta first, then calculate new dx and dy
    dtheta = (V * math.sin(phi)) / config.L if abs(phi) > 1e-6 else 0 # Ensure no division by zero
    theta += dtheta * dt
    dx = V * math.cos(theta) * math.cos(phi)
    dy = V * math.sin(theta) * math.cos(phi)

    # Update position
    car_position_road[0] += dx * dt
    car_position_road[1] += dy * dt

    car_position_screen[0] = car_position_road[0]
    
    return car_position_road, car_position_screen, theta, phi

def get_velocity_y_direction(V, theta, phi):
    return V * math.sin(theta) * math.cos(phi)

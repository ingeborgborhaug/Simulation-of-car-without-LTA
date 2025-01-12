import math
import numpy as np
import pygame

# General 
PIXEL_TO_METER = 0.034
METER_TO_PIXEL = 1 / PIXEL_TO_METER
SIMULATION_TIME = 120 # seconds

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = None

# Lane positions (initially)
LANE_LEFT = SCREEN_WIDTH/3 # Offset from the left of the screen
LANE_WIDTH = 6.5 * METER_TO_PIXEL # Width of the lane is typically 3.7 meters  #ADRIAN: 5
LANE_RIGHT = LANE_LEFT + 2*LANE_WIDTH  
LANE_CENTER = LANE_LEFT + LANE_WIDTH 

# Noise and missing-point parameters
NOISE_STD = 1.5  # Standard deviation for Gaussian noise (in pixels)
NAN_PROBABILITY = 0.01  # 20% chance of a point being replaced with np.nan


# New car parameters
CAR_WIDTH = 50 # Width of the car is typically 1.7meters, according to this, 1 pixel is 0.034 meters
CAR_LENGTH =  CAR_WIDTH * 2.65 # Length of the car is typically 4.5 meters

OPTIMAL_DISTANCE = (LANE_WIDTH - CAR_WIDTH)/2 
CAR_POSITION_ROAD_INIT = [LANE_CENTER + LANE_WIDTH/2, SCREEN_HEIGHT - 150]
CAR_POSITION_SCREEN_INIT = [LANE_CENTER + LANE_WIDTH/2, SCREEN_HEIGHT - 150]

K_PHI = 5  # Proportional gain for phi adjustment
DAMPING_FACTOR = 0.8

# Load the car image (global or in setup)
CAR_IMAGE = pygame.image.load("LAB_2_delivery/car_img.png")
CAR_IMAGE = pygame.transform.scale(CAR_IMAGE, (CAR_WIDTH*1, CAR_LENGTH*1))  # Scale to match car dimensions
CAR_IMAGE = pygame.transform.rotate(CAR_IMAGE, -90)

THETA_INIT = -np.pi/2                  # Orientation of the car (radians)
L = 50                     # Wheelbase (distance between axles)
V = 40                      # Linear velocity
PHI_INIT = 0                    # Steering angle
DPHI_INIT = 0
MAX_PHI = math.radians(20)  # Limit steering angle #KRISTINE: 45
MAX_DPHI = 0.6
DELTA_DPHI = V*0.01 # How much angle changes per key press
dt = 0.1   

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)
CYAN = (0, 255,255)
GREEN = (0,255,0)

# LTA
T = 3
LTA_TOLERANCE =  0.35*METER_TO_PIXEL    #  0.35*METER_TO_PIXEL 
SENSOR_LOOKAHEAD = 8
W_i_x = 1400
W_i_THETA = 200
W_i_u = 0.1

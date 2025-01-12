import pygame
import config

def controller(keys, phi, recovering_flag, status):

    dphi_human = 0
    dphi_output = 0
    dphi_damped = 0

    if keys[pygame.K_LEFT]and status not in ["crossing_LTA_threshold_left"]:
        dphi_human = -config.DELTA_DPHI
        dphi_output = dphi_human

    elif keys[pygame.K_RIGHT] and status not in ["crossing_LTA_threshold_right"]:
        dphi_human = config.DELTA_DPHI
        dphi_output = dphi_human

    elif not recovering_flag:
        if abs(phi) < 1e-2: # Smoothly reduce phi to 0 (simulate wheel returning to neutral)
            phi = 0
            dphi_damped = 0
            dphi_output = dphi_damped
        else:
            dphi_damped = -config.K_PHI * phi
            dphi_damped *= config.DAMPING_FACTOR # Apply damping to slow down convergence and prevent oscillations
            dphi_output = dphi_damped


    return phi, dphi_output, dphi_human

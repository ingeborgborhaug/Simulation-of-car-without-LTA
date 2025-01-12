import config
import numpy as np
import scipy
import matplotlib.pyplot as plt
import navigation

# from documentation import plot_ref_against_pred_position
def get_guidance_input(car_position_road, car_position_screen, theta, phi, dphi, LTA_left_detection_processed, LTA_right_detection_processed, status):
    
    lta_dphi = mpc_control(car_position_road, car_position_screen[1], theta, phi, dphi, LTA_left_detection_processed, LTA_right_detection_processed, status)
    lta_dphi = max(min(lta_dphi, config.MAX_DPHI), -config.MAX_DPHI)

    return lta_dphi

def mpc_control(car_position_road, car_y_screen, theta, phi, dphi, LTA_left_detection_processed, LTA_right_detection_processed, status):
    """
    Solve the MPC optimization problem.
    :param x_initial: Current state [e_lat, e_head, e_lat_dot, e_head_dot]
    :param ref_traj: Reference trajectory (target lateral error and heading error)
    :param dt: Time step
    :return: Optimal steering input (dphi, V)
    """
    # Get optimal path
    optimal_trajectory, optimal_theta  = navigation.get_optimal_trajectory(LTA_left_detection_processed, LTA_right_detection_processed)
    navigation.plot_trajectory(optimal_trajectory, optimal_theta, config.BLUE)

    # Get recovery path
    recovery_trajectory, recovery_theta = navigation.recovery_trajectory_spline(car_position_road, car_y_screen,theta, phi, config.dt, config.T, config.V, optimal_theta, optimal_trajectory)
    x_ref = [x for x, y in recovery_trajectory]
    navigation.plot_trajectory(recovery_trajectory, recovery_theta, config.RED)

    '''
    if recovering_flag and status in ["outside_left", "near_left"]:
        recovery_trajectory, recovery_theta = recovery_trajectory_spline_k(car_position_road, car_y_screen, theta, phi, config.dt, config.T, config.V, optimal_theta, optimal_trajectory, lookahead=200, left_boundary=True)        
    elif recovering_flag and status in ["outside_right", "near_right"]:
        recovery_trajectory, recovery_theta = recovery_trajectory_spline_k(car_position_road, car_y_screen, theta, phi, config.dt, config.T, config.V, optimal_theta, optimal_trajectory, lookahead=200, left_boundary=False)        
    '''        
    
    horizon = len(recovery_trajectory)
    
    # Get predicted path
    predicted_trajectory, predicted_theta = navigation.predict_trajectory(car_position_road, car_y_screen, theta, phi, dphi, horizon)
    x_pred = [x for x, y in predicted_trajectory]
    navigation.plot_trajectory(predicted_trajectory, predicted_theta, config.YELLOW)

    #---------TESTING---------
    # x_preds = []
    # time = [i * config.dt for i in range(horizon)]
    # plot_ref_against_pred_positions(x_ref, x_pred, time)
    # -------------------------

    if status in "outside_left":
        dphi_control_initial_guess = np.ones(horizon) * (config.MAX_DPHI) 
    elif status in "near_left":
        dphi_control_initial_guess = np.ones(horizon) * (config.MAX_DPHI/2)
    elif status in "outside_right":
        dphi_control_initial_guess = np.ones(horizon) * (-config.MAX_DPHI)
    elif status in "near_right":
        dphi_control_initial_guess = np.ones(horizon) * (-config.MAX_DPHI/2) 
        
        
    #TODO: Improove initial guess by inverting the last x-numbers of DPHI
    # bounds = [(-np.radians(config.MAX_DPHI), np.radians(config.MAX_DPHI))] * horizon

    # Compare optimal and predicted path
    res = scipy.optimize.minimize(
        cost,                                                           # The cost function to minimize
        dphi_control_initial_guess,                                     # Initial guess for the steering input sequence
        args=(x_pred, predicted_theta, x_ref, recovery_theta, horizon), # Additional arguments required by the cost function
        method="SLSQP",                                                 # Sequential Least Squares Programming
        # bounds=bounds,
        options=dict(maxiter=100)
    )

    optimal_dphi_control = res.x[0] 
    print('optimal_dphi_control', optimal_dphi_control)
    text_optimal_steering = config.FONT.render(f"Phi from MPC: {optimal_dphi_control:.2f}", True, config.WHITE)        # Top-left corner
    config.SCREEN.blit(text_optimal_steering, (10, 130))          # Below x

    # if res.success:
    #     print("Optimization succeeded!")
    #     print("Optimized control inputs (phi_control):", res.x)
    #     print("Final cost value:", res.fun)
    # else:
    #     print("Optimization failed.")
    #     print("Reason:", res.message)

    if status in ["outside_left", "near_left"]:
        optimal_dphi_control = abs(optimal_dphi_control)
    elif status in ["outside_right", "near_right"]:
        optimal_dphi_control = - abs(optimal_dphi_control)


    return optimal_dphi_control # dphi_control

def cost(dphi, x_array, theta_array, x_car_ref, theta_ref, horizon):
    """
    dphi_control[i] <- steering angle for step i

    """
    x_array = np.array(x_array)
    x_car_ref = np.array(x_car_ref)
    theta_array = np.array(theta_array)
    theta_ref = np.array(theta_ref)

    cost_val = 0.0
    w_i_x = config.W_i_x
    w_i_theta = config.W_i_THETA
    w_i_u = config.W_i_u

    for i in range(horizon):
        cost_val += ( (w_i_x * (x_array[i] - x_car_ref[i]) ** 2 + w_i_theta * (theta_array[i] - theta_ref[i]) ** 2) + w_i_u * dphi[i] ** 2 ) 
    return cost_val



# -*- coding: utf-8 -*-

import numpy as np

BASE_STATION_IP = "10.11.58.50"
TERMINAL_IP = "10.11.58.53"

carrier_frequency = 6.08e9
sampling_rate = 4.9152e8
max_receive_delay = 1e-6
num_prepended_zeros=3000
num_appended_zeros=3000
num_drops = 400

base_station_x_distance = -0.061
base_station_y_distance = 0.062
base_station_tx_antenna_positions = np.array([
    [0 * base_station_x_distance, 0, 0],
    [1 * base_station_x_distance, 0, 0],
    [2 * base_station_x_distance, 0, 0],
], dtype=np.float64)

base_station_rx_antenna_positions = np.array([
    [0.5 * base_station_x_distance, base_station_y_distance, 0],
    [1.5 * base_station_x_distance, base_station_y_distance, 0],
    [2.5 * base_station_x_distance, base_station_y_distance, 0],
], dtype=np.float64)

terminal_tx_antenna_positions = np.array([
    [0.0, 0, 0]
], dtype=np.float64)

terminal_rx_antenna_positions = np.array([
    [-0.062, 0.0, 0.0],
], dtype=np.float64)

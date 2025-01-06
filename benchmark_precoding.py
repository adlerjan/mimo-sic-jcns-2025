# -*- coding: utf-8 -*-
#
# This script benchmarks the performance of the precoding algorithm
# proposed in the paper "Self-Interference Cancellation in Digital Sensing and Communication Arrays",
# to be presented at the 2025 5th IEEE International Symposium on Joint Communications & Sensing.
#
# Please refer to the README.md file for information on how to install the required dependencies.
# For additional questions, please don't hesitate to contact jan.adler@barkhauseninstitut.org .

from __future__ import annotations
from typing import Literal

import numpy as np
from h5py import File
from scipy.constants import speed_of_light
from rich.prompt import Prompt

from hermespy.core import Transformation
from hermespy.core.evaluators import ReceivePowerEvaluator, TransmitPowerEvaluator
from hermespy.beamforming import SphericalFocus
from hermespy.hardware_loop import HardwareLoop, DeviceReceptionPlot, RadarRangePlot, ReceivedConstellationPlot, DeviceTransmissionPlot, ArtifactPlot, ScalarAntennaCalibration
from hermespy.hardware_loop.uhd import UsrpDevice, UsrpSystem
from hermespy.modem import ReceivingModem, ConstellationEVM

from probe import probe_mmse, probe_noise, probe_delay
from dsp import NewJCAS, LeakingPowerEvaluator
from parameters import *
from waveforms import ofdm

# Additional global parameters
half_wavlength = speed_of_light / (2 * carrier_frequency)
visualize = False

scenario = UsrpSystem()
loop = HardwareLoop[UsrpSystem, UsrpDevice](scenario)
loop.num_drops = num_drops
loop.plot_information = False

jcas_device = scenario.new_device(
    ip=BASE_STATION_IP,
    carrier_frequency=carrier_frequency,
    sampling_rate=sampling_rate,
    selected_transmit_ports=[0, 1, 2],
    selected_receive_ports=[1, 2],
    tx_gain=55.0,
    rx_gain=30.0,
    max_receive_delay=max_receive_delay,
    num_prepended_zeros=num_prepended_zeros,
    num_appended_zeros=num_appended_zeros,
    scale_transmission=False,
)

terminal_device = scenario.new_device(
    ip=TERMINAL_IP,
    carrier_frequency=carrier_frequency,
    sampling_rate=sampling_rate,
    selected_transmit_ports=[0],
    selected_receive_ports=[0],
    tx_gain=55.0,
    rx_gain=30.0,
    max_receive_delay=max_receive_delay,
    num_prepended_zeros=num_prepended_zeros,
    num_appended_zeros=num_appended_zeros,
    scale_transmission=False,
)

# Configure the global positions of both devices
jcas_device.pose = Transformation.From_Translation(np.array([0, 0, 0], dtype=np.float64))
terminal_device.pose = Transformation.From_Translation(np.array([0, 0, 2.0], dtype=np.float64))
terminal_device.pose.rotation_rpy = np.array([0, 0, np.pi], dtype=np.float64)

# Configure antenna array topologies
for antenna_position, port in zip(base_station_tx_antenna_positions, jcas_device.antennas.transmit_ports):
    port.pose = Transformation.From_Translation(antenna_position)
for antenna_position, port in zip(base_station_rx_antenna_positions, jcas_device.antennas.receive_ports):
    port.pose = Transformation.From_Translation(antenna_position)
for antenna_position, port in zip(terminal_tx_antenna_positions, terminal_device.antennas.transmit_ports):
    port.pose = Transformation.From_Translation(antenna_position)
for antenna_position, port in zip(terminal_rx_antenna_positions, terminal_device.antennas.receive_ports):
    port.pose = Transformation.From_Translation(antenna_position)

# Query the regularization
regularization = float(Prompt.ask('Regularization', console=loop.console, default=1e-11))

# Base station DSP chain
jcas_dsp = NewJCAS(
    max_range=3,
    waveform=ofdm,
    beam_focus=SphericalFocus(0.0, 0.0),
    normalize='amplitude',
    regularization=regularization,
)
jcas_device.add_dsp(jcas_dsp)

# Terminal DSP chain
terminal_dsp = ReceivingModem()
terminal_dsp.waveform = ofdm
terminal_device.add_dsp(terminal_dsp)

# Configure evaluators
analog_leakage = LeakingPowerEvaluator(jcas_dsp, 'analog')
digital_leakage = LeakingPowerEvaluator(jcas_dsp, 'digital')
evm = ConstellationEVM(jcas_dsp, terminal_dsp)
tx_jcas_power = TransmitPowerEvaluator(jcas_dsp)
rx_jcas_power = ReceivePowerEvaluator(jcas_dsp)
rx_terminal_power = ReceivePowerEvaluator(terminal_dsp)
loop.add_evaluator(analog_leakage)
loop.add_evaluator(digital_leakage)
loop.add_evaluator(evm)
loop.add_evaluator(tx_jcas_power)
loop.add_evaluator(rx_jcas_power)
loop.add_evaluator(rx_terminal_power)

noise_estimate = probe_noise(
    jcas_device,
    num_probes=10,
    num_wavelet_samples=jcas_dsp.num_samples,
    console=loop.console,
)

jcas_device.delay_calibration = probe_delay(
    jcas_device,
    num_probes=10,
    console=loop.console,
)

jcas_device.antenna_calibration = ScalarAntennaCalibration.Estimate(
    loop.scenario,
    jcas_device,
    terminal_device,
)

leakage_estimate = probe_mmse(
    jcas_device,
    noise_estimate,
    num_probes=7,
    num_wavelet_samples=jcas_dsp.num_samples,
)

# Reset scenario
jcas_device.max_receive_delay = .25 * max_receive_delay
terminal_device.max_receive_delay = .25 * max_receive_delay
scenario.seed = 42

state: Literal['calibrated', 'uncalibrated', 'test-calibrated', 'test-uncalibrated'] = Prompt.ask('', console=loop.console, choices=['calibrated', 'uncalibrated', 'test-calibrated', 'test-uncalibrated'])

if state == 'uncalibrated' or state == 'test-uncalibrated':
    # Generate a single drop without calibration
    jcas_dsp.interference_estimate = np.zeros((jcas_device.num_digital_receive_ports, jcas_device.num_digital_transmit_ports, jcas_dsp.num_samples), dtype=np.complex128)

elif state == 'calibrated' or state == 'test-calibrated':
    # Generate a single drop with calibration
    jcas_dsp.interference_estimate = leakage_estimate.leakage_response
    scenario.seed = 42

else:
    loop.console.print('Invalid state')
    exit(-1)

if visualize or state.startswith('test'):
    loop.plot_information = True
    loop.add_plot(DeviceReceptionPlot(terminal_device, "Terminal Rx"))
    loop.add_plot(RadarRangePlot(jcas_dsp))
    loop.add_plot(ReceivedConstellationPlot(terminal_dsp))
    loop.add_plot(DeviceReceptionPlot(jcas_device, "Base Rx"))
    loop.add_plot(DeviceTransmissionPlot(jcas_device, "Base Tx"))
    loop.add_plot(ArtifactPlot(evm))

if state.startswith('test'):
    loop.num_drops = 10000
    loop.run()

else:

    experiment = state + '_0' if regularization == 0 else state + '_' + str(-int(np.log10(regularization)))
    loop.results_dir = loop.default_results_dir(experiment=experiment, overwrite_results=True)

    # Store the self-interference estimate
    file = File(loop.results_dir + '/leakage_estimate.h5', 'w')
    leakage_estimate.to_HDF(file.create_group('leakage_estimate'))
    file.close()

    Prompt.ask('Add target', console=loop.console)
    loop.run(campaign='h1', serialize_state=False, overwrite=True)

    Prompt.ask('Remove target', console=loop.console)
    loop.run(campaign='h0', serialize_state=False, overwrite=False)

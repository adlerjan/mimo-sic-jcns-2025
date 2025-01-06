# -*- coding: utf-8 -*-
#
# This script generates Figure 8: Detection performance of the paper "Self-Interference Cancellation in Digital Sensing and Communication Arrays",
# to be presented at the 2025 5th IEEE International Symposium on Joint Communications & Sensing.
# It requires the results generated by the benchmark precoding script.
#
# Please refer to the README.md file for information on how to install the required dependencies.
# For additional questions, please don't hesitate to contact jan.adler@barkhauseninstitut.org .

from os import path as path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # type: ignore
from h5py import File

from beamforming import SphericalFocus
from hermespy.radar import ReceiverOperatingCharacteristic
from hermespy.hardware_loop import SelectiveLeakageCalibration
from hermespy.hardware_loop.uhd import UsrpSystem
from hermespy.modem import ReceivingModem

from dsp import NewJCAS
from parameters import *
from waveforms import ofdm


# Set the correct matplotlib style for IEEE papers
plt.style.use(['science', 'ieee', 'no-latex'])

scenario = UsrpSystem()

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

# Base station DSP chain
jcas_dsp = NewJCAS(
    max_range=3,
    waveform=ofdm,
    beam_focus=SphericalFocus(0.0, 0.0),
    normalize='amplitude',
)
jcas_device.add_dsp(jcas_dsp)

# Terminal DSP chain
terminal_dsp = ReceivingModem()
terminal_dsp.waveform = ofdm
terminal_device.add_dsp(terminal_dsp)

results_path = path.join(path.dirname(__file__), '..', 'results')

def load_roc_calibrated(campaign: str):
    file = File(path.join(results_path, campaign, 'leakage_estimate.h5'), 'r')
    calibration = SelectiveLeakageCalibration.from_HDF(file['leakage_estimate'])
    file.close()
    jcas_dsp.interference_estimate = calibration.leakage_response

    # Hack: Make a single transmission to update the precoding
    jcas_dsp.transmit(jcas_device.state(), notify=False)

    scenario.replay('results/' + campaign + '/drops.h5', campaign='h0')
    roc = ReceiverOperatingCharacteristic.FromScenario(scenario, jcas_dsp)
    return roc


def load_roc_uncalibrated():
    jcas_dsp.interference_estimate = np.zeros((jcas_device.num_digital_receive_ports, jcas_device.num_digital_transmit_ports, jcas_dsp.num_samples))

    # Hack: Make a single transmission to update the precoding
    jcas_dsp.transmit(jcas_device.state(), notify=False)

    scenario.replay(path.join(results_path, 'uncalibrated_10', 'drops.h5'), campaign='h0')
    roc = ReceiverOperatingCharacteristic.FromScenario(scenario, jcas_dsp)
    return roc


# Load ROC curves
curves = {
    'Uncoded': load_roc_uncalibrated(),
    r'$\lambda = 1$': load_roc_calibrated('calibrated_0'),
    r'$\lambda = 10^{-10}$': load_roc_calibrated('calibrated_10'),
    r'$\lambda = 10^{-18}$': load_roc_calibrated('calibrated_18'),
}

figure, axes = plt.subplots(squeeze=True, tight_layout=True)
for label, roc in curves.items():
    values = roc.to_array()
    pd = values[0, :, 1]
    pfa = values[0, :, 0]
    axes.plot(pd, pfa, label=label)

axes.legend(loc='lower right', prop={'size': 6})
axes.set_xlabel('False Alarm Probability')
axes.set_ylabel('Detection Probability')
axes.set_xlim(-.05, 1.0)
axes.set_ylim(0.0, 1.1)

plt.show()

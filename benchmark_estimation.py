# -*- coding: utf-8 -*-
#
# This script benchmarks the performance of the leakage estimation algorithm
# proposed in the paper "Self-Interference Cancellation in Digital Sensing and Communication Arrays",
# to be presented at the 2025 5th IEEE International Symposium on Joint Communications & Sensing.
#
# Please refer to the README.md file for information on how to install the required dependencies.
# For additional questions, please don't hesitate to contact jan.adler@barkhauseninstitut.org .

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from scipy.constants import speed_of_light
from rich.console import Console

from hermespy.core import Artifact, Evaluator, Evaluation, RandomNode, DenseSignal, ScalarEvaluationResult
from hermespy.core.monte_carlo import Evaluation, GridDimension
from hermespy.hardware_loop import HardwareLoop, IterationPriority, NoDelayCalibration, PhysicalDevice
from hermespy.hardware_loop.uhd import UsrpDevice, UsrpSystem

from probe import probe_delay, probe_mmse, probe_noise
from parameters import *

# Additional global parameters
half_wavlength = speed_of_light / (2 * carrier_frequency)
num_samples = 127
num_drops = 1  # override default parameters

scenario = UsrpSystem()
loop = HardwareLoop[UsrpSystem, UsrpDevice](scenario)
loop.num_drops = num_drops
loop.plot_information = False

# Add JCAS device
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


class LeakageCalibrationArtifact(Artifact):
    __rmse: float

    def __init__(self, rmse: float) -> None:
        self.__rmse = rmse

    @property
    def rmse(self) -> float:
        return self.__rmse

    def __str__(self) -> str:
        return f"{self.rmse:.2f}"

    def to_scalar(self) -> float:
        return self.rmse


class LeakageCalibrationEvaluation(Evaluation):

    __rmse: float

    def __init__(self, rmse: float) -> None:
        self.__rmse = rmse

    @property
    def rmse(self) -> float:
        return self.__rmse

    def artifact(self) -> LeakageCalibrationArtifact:
        return LeakageCalibrationArtifact(self.rmse)

    def _prepare_visualization(self, figure, axes, **kwargs):
        pass

    def _update_visualization(self, visualization, **kwargs):
        pass


class LeakageCalibrationEvaluator(Evaluator, RandomNode):
    __device: PhysicalDevice
    __num_samples: int

    def __init__(self, device: PhysicalDevice, num_samples: int) -> None:

        # Initialize base classes
        Evaluator.__init__(self)
        RandomNode.__init__(self)

        # Initialize attributes
        self.__device = device
        self.__num_samples = num_samples

    @property
    def device(self) -> PhysicalDevice:
        return self.__device

    @property
    def num_samples(self) -> int:
        return self.__num_samples

    def evaluate(self) -> Evaluation:
        # Transmit a white noise signal
        evaluation_samples = self._rng.standard_normal((self.device.num_digital_transmit_ports, self.num_samples)) + 1j * self._rng.standard_normal((self.device.num_digital_transmit_ports, self.num_samples))
        evaluation_samples /= max(np.abs(evaluation_samples.imag).max(), np.abs(evaluation_samples.real).max())
        burst_signal = DenseSignal(evaluation_samples, self.device.sampling_rate, self.device.carrier_frequency)

        received_signal = self.device.trigger_direct(burst_signal, calibrate=False)
        predicted_signal = self.device.leakage_calibration.predict_leakage(burst_signal)
        rmse = np.sqrt(np.mean(np.abs(received_signal.getitem((slice(None), slice(0, predicted_signal.num_samples))) - predicted_signal.getitem()) ** 2))

        # Compute rmse between predicted and received signal
        return LeakageCalibrationEvaluation(rmse)

    @property
    def abbreviation(self) -> str:
        return "LC"

    @property
    def title(self) -> str:
        return "Leakage Calibration"

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self, False)


class NoisePowerArtifact(Artifact):
    __noise_power: float

    def __init__(self, noise_power: float) -> None:
        self.__noise_power = noise_power

    @property
    def noise_power(self) -> float:
        return self.__noise_power

    def __str__(self) -> str:
        return f"{self.noise_power:.2f}"

    def to_scalar(self) -> float:
        return self.noise_power


class NoisePowerEvaluation(Evaluation):
        __noise_power: float

        def __init__(self, noise_power: float) -> None:
            self.__noise_power = noise_power

        @property
        def noise_power(self) -> float:
            return self.__noise_power

        def artifact(self) -> NoisePowerArtifact:
            return NoisePowerArtifact(self.noise_power)

        def _prepare_visualization(self, figure, axes, **kwargs):
            pass

        def _update_visualization(self, visualization, **kwargs):
            pass


class NoisePowerEvaluator(Evaluator):

    __device: PhysicalDevice

    def __init__(self, device: PhysicalDevice) -> None:

        # Initialize base classes
        Evaluator.__init__(self)

        # Initialize attributes
        self.__device = device

    @property
    def device(self) -> PhysicalDevice:
        return self.__device

    def evaluate(self) -> Evaluation:
        return NoisePowerEvaluation(np.mean(self.device.noise_power, keepdims=False))

    @property
    def abbreviation(self) -> str:
        return "NP"

    @property
    def title(self) -> str:
        return "Noise Power"

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self, False)


def recalibrate_hook(system: UsrpSystem, console: Console):

    noise_estimate = probe_noise(
        jcas_device,
        num_probes=10,
        num_wavelet_samples=num_samples,
        console=console,
    )
    jcas_device.noise_power = noise_estimate

    jcas_device.delay_calibration = NoDelayCalibration()
    jcas_device.delay_calibration = probe_delay(
        jcas_device,
        10,
        console=console,
    )

    jcas_device.leakage_calibration = probe_mmse(
        jcas_device,
        noise_estimate,
        num_probes=7,
        num_wavelet_samples=num_samples,
    )


loop.add_pre_drop_hook(recalibrate_hook)
loop.add_evaluator(LeakageCalibrationEvaluator(jcas_device, 1024))
loop.add_evaluator(NoisePowerEvaluator(jcas_device))
loop.new_dimension('rx_gain', np.linspace(0, 60, 13, endpoint=True), jcas_device)
loop.iteration_priority = IterationPriority.DROPS
loop.results_dir = loop.default_results_dir('check_calibration', overwrite_results=True)

loop.run(serialize_state=False)

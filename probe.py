# coding: utf-8

from __future__ import annotations

import numpy as np
from numpy.linalg import svd
from scipy.fft import fft, ifft
from rich import get_console
from rich.console import Console

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SimulationScenario
from hermespy.hardware_loop import DelayCalibration, PhysicalDevice, PhysicalScenario, SelectiveLeakageCalibration


def probe_noise(
    device: PhysicalDevice | SimulatedDevice,
    num_probes: int,
    num_wavelet_samples: int,
    console: Console | None = None,
) -> np.ndarray:
    _console = get_console() if console is None else console
    _console.print(f"Estimating noise statistics for a device with {device.num_digital_receive_ports} receive ports")
    
    # Signal transmitter transmitting zeros over the device's dacs
    empty_signal = Signal.Create(
        np.zeros((device.num_digital_transmit_ports, num_wavelet_samples), dtype=np.complex128),
        sampling_rate=device.sampling_rate,
        carrier_frequency=device.carrier_frequency,
    )

    received_powers = np.zeros((num_probes, device.num_digital_receive_ports), dtype=np.float64)

    for p in range(num_probes):
        received_powers[p, :] = device.trigger_direct(empty_signal, calibrate=False).power


    noise_powers = np.mean(received_powers, axis=0)
    _console.print(f"Mean noise power: {float(np.mean(noise_powers))}")
    return noise_powers

def probe_delay(
    device: PhysicalDevice | SimulatedDevice,
    num_probes: int,
    console: Console | None = None,
) -> DelayCalibration:
    _console = get_console() if console is None else console
    _console.print(f"Estimating the delay for a device with {device.num_digital_receive_ports} receive ports")
    
    num_samples = int(1 + device.max_receive_delay * device.sampling_rate)
    tx_samples = np.zeros((device.num_digital_transmit_ports, num_samples), dtype=np.complex128)
    tx_samples[0, 0] = 1.0
    rx_samples = np.zeros((device.num_digital_receive_ports, num_samples), dtype=np.complex128)
    
    # Signal transmitter transmitting zeros over the device's dacs
    tx_signal = Signal.Create(
        tx_samples,
        sampling_rate=device.sampling_rate,
        carrier_frequency=device.carrier_frequency,
    )
    
    for _ in range(num_probes):
        rx_samples += 1 / num_probes * device.trigger_direct(tx_signal).getitem()[:, :num_samples]
        
    delay_estimation = np.mean(np.abs(rx_samples), axis=0)
    delay = np.argmax(delay_estimation) / device.sampling_rate
    _console.print(f"Estimated delay: {delay} s")
    return DelayCalibration(delay, device)
    
def probe_leakage(
    device: PhysicalDevice | SimulatedDevice,
    num_probes: int,
    num_wavelet_samples: int,
) -> tuple[np.ndarray, np.ndarray]:

    # Generate zadoff-chu sequences to probe the device leakage
    cf = num_wavelet_samples % 2
    q = 1
    sample_indices = np.arange(num_wavelet_samples)
    probe_indices = np.arange(1, 1 + num_probes)
    zadoff_chu_sequences = np.exp(
        -1j
        * np.pi
        * np.outer(probe_indices, sample_indices * (sample_indices + cf + 2 * q))
        / num_wavelet_samples
    )

    # Replicate the (periodic) ZC waveform such that any window of
    # num_wavelet_samples will always contain an entire ZC sequence
    # (time-shifted). Essentially, build a huge CP and CS around the center
    # ZC sequence. At the receiver, we will focus on receiving the second
    # (i.e. center) ZC waveform.
    probing_waveforms = np.tile(zadoff_chu_sequences, (1, 3))
    probing_frequencies = fft(zadoff_chu_sequences, axis=1, norm="ortho")
    num_samples = probing_waveforms.shape[1]

    # Collect received samples
    received_waveforms = np.zeros(
        (
            num_probes,
            device.antennas.num_receive_ports,
            device.antennas.num_transmit_ports,
            num_wavelet_samples,
        ),
        dtype=np.complex_,
    )
    for p, n in np.ndindex(num_probes, device.antennas.num_transmit_ports):
        tx_samples = np.zeros(
            (device.antennas.num_transmit_ports, num_samples), dtype=np.complex_
        )
        tx_samples[n, :] = probing_waveforms[p, :]
        rx_signal = device.trigger_direct(Signal.Create(tx_samples, device.sampling_rate, device.carrier_frequency), calibrate=False)

        # From the received signal, collect the middle num_wavelet_samples of the transmitted
        # ZC sequences. This should account for any delays of the transmission, as long as the
        # sequence is long enough. If it's not long enough, the leakage calculation will fail.
        # TODO: look at the estimated delay and its reliability. If the delay cannot be estimated
        # reliably, most probably, the TX signal was not received in the window decided here
        start = num_wavelet_samples
        received_waveforms[p, :, n, :] = rx_signal.getitem(
            (slice(None, None), slice(start, start + num_wavelet_samples))
        )

    # Compute received frequency spectra
    received_frequencies = fft(received_waveforms, axis=3, norm="ortho")

    # Return the collected probing and received frequency spectra
    return probing_frequencies, received_frequencies


def probe_least_squares_first_version(
    device: PhysicalDevice | SimulatedDevice,
    scenario: PhysicalScenario | SimulationScenario,
    num_probes: int = 7,
    num_wavelet_samples: int = 4673,
) -> SelectiveLeakageCalibration:
    """Estimate the transmit-receive leakage for a device using Leat-Squares estimation.

    Args:

        device (PhysicalDevice | SimulatedDevice):
            Device to estimate the leakage calibration for.
            
        scenario (PhysicalScenario | SimulationScenario):
            Scenario containing the respective device.

        num_probes (int, optional):
            Number of probings transmitted to estimate the covariance matrix.
            :math:`7` by default.

        num_wavelet_samples (int, optional):
            Number of samples transmitted per probing to estimate the covariance matrix.
            :math:`4673` by default.

    Returns: The initialized :class:`SelectiveLeakageCalibration` instance.

    Raises:

        ValueError: If the number of probes is not strictly positive.
        ValueError: If the number of samples is not strictly positive.
    """

    # Probe the device leakage
    probing_frequencies, received_frequencies = probe_leakage(device, scenario, num_probes, num_wavelet_samples)
    num_samples = probing_frequencies.shape[1]

    estimated_frequency_response = np.zeros(
        (device.antennas.num_receive_ports, device.antennas.num_transmit_ports, num_samples),
        dtype=np.complex_,
    )
    for m, n in np.ndindex(
        device.antennas.num_receive_ports, device.antennas.num_transmit_ports
    ):
        # Select the transmitted and received frequency spectra for the current antenna pairs
        rx_frequencies = received_frequencies[:, m, n, :]
        tx_frequencies = probing_frequencies[:, :]

        # Estimate the frequency-selectivity by least-squares estimation
        Rx = rx_frequencies
        Tx = tx_frequencies

        # Solve for X (i.e. the channel frequency response) in the least-squares sense:
        # Minimize \|Rx - X*Tx\|^2 under the constraint that X is diagonal.
        # See https://math.stackexchange.com/a/3502842/397295
        # results in this expression:
        # x_ls = np.diag(Tx.conj().T.dot(Rx)) / np.diag(Tx.conj().T.dot(Tx))

        # optimized version:
        x_ls = np.sum(Tx.conj() * Rx, axis=0) / np.sum(Tx.conj() * Tx, axis=0)

        estimated_frequency_response[m, n, :] = x_ls

    # Convert the estimated leakage-response into the time-domain
    leakage_response = ifft(estimated_frequency_response, norm="backward")

    calibration_delay = 0.0
    if isinstance(device, PhysicalDevice):
        calibration_delay = device.delay_calibration.delay

    calibration = SelectiveLeakageCalibration(
        leakage_response, device.sampling_rate, calibration_delay
    )

    return calibration


def probe_mmse(
    device: PhysicalDevice | SimulatedDevice,
    noise_power: np.ndarray,
    num_probes: int = 7,
    num_wavelet_samples: int = 4673,
) -> SelectiveLeakageCalibration:
    """Estimate the transmit receive leakage for a physical device using Minimum Mean Square Error (MMSE) estimation.

    Args:

        device (PhysicalDevice):
            Physical device to estimate the covariance matrix for.

        num_probes (int, optional):
            Number of probings transmitted to estimate the covariance matrix.
            :math:`7` by default.

        num_wavelet_samples (int, optional):
            Number of samples transmitted per probing to estimate the covariance matrix.
            :math:`127` by default.

        noise_power (np.ndarray, optional):
            Noise power at the receiving antennas.
            If not specified, the device's noise power configuration will be assumed or estimated on-the-fly.

    Returns: The initialized :class:`SelectiveLeakageCalibration` instance.

    Raises:

        ValueError: If the number of probes is not strictly positive.
        ValueError: If the number of samples is not strictly positive.
        ValueError: If the noise power is negative.
    """

    # Probe the device leakage
    probing_frequencies, received_frequencies = probe_leakage(device, num_probes, num_wavelet_samples)
    num_samples = probing_frequencies.shape[1]

    if np.any(noise_power < 0.0):
        raise ValueError(f"Noise power must be non-negative (not {noise_power})")

    if noise_power.ndim != 1 or noise_power.shape[0] != device.antennas.num_receive_ports:
        raise ValueError("Noise power has invalid dimensions")

    # Estimate frequency spectra via MMSE estimation
    # https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture20.pdf
    # Page 2 Example 1

    h = np.zeros((num_samples * num_probes, num_samples), dtype=np.complex_)
    for p, probe in enumerate(probing_frequencies):
        h[p * num_samples : (1 + p) * num_samples, :] = np.diag(probe)

    # For now, the noise power is assumed to be the mean over all receive chains
    mean_noise_power = np.mean(noise_power)

    # Compute the pseudo-inverse of the received frequency spectra by singular value decomposition
    u, s, vh = svd(
        h @ h.T.conj()
        + mean_noise_power * np.eye(num_samples * num_probes, num_samples * num_probes),
        full_matrices=False,
        hermitian=True,
    )
    u_select = u[:, :num_samples]
    s_select = s[:num_samples]
    vh_select = vh[:num_samples, :]

    mmse_estimator = h.T.conj() @ vh_select.T.conj() @ np.diag(1 / s_select) @ u_select.T.conj()

    # Estimate the frequency spectra for each antenna probing independently
    mmse_frequency_selectivity_estimation = np.zeros(
        (device.antennas.num_receive_ports, device.antennas.num_transmit_ports, num_samples),
        dtype=np.complex_,
    )
    for m, n in np.ndindex(
        device.antennas.num_receive_ports, device.antennas.num_transmit_ports
    ):
        probing_estimation = mmse_estimator @ received_frequencies[:, m, n, :].flatten()
        mmse_frequency_selectivity_estimation[m, n, :] = probing_estimation

    # Initialize the calibration from the estimated frequency spectra
    leakage_response = ifft(mmse_frequency_selectivity_estimation, axis=2)[
        :, :, :num_wavelet_samples
    ]

    calibration_delay = 0.0
    if isinstance(device, PhysicalDevice):
        calibration_delay = device.delay_calibration.delay

    calibration = SelectiveLeakageCalibration(
        leakage_response, device.sampling_rate, calibration_delay
    )
    return calibration

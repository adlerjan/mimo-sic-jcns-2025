# -*- coding: utf-8 -*-

from __future__ import annotations
from collections.abc import Sequence
from math import ceil
from typing import Literal, Sequence

from h5py import Group
import numpy as np
from scipy.constants import speed_of_light
from scipy.signal import correlate, correlation_lags
from scipy.fft import fft, ifft

from hermespy.beamforming import BeamFocus, SphericalFocus
from hermespy.core import AntennaMode, Device, Signal, Evaluator, Evaluation, Artifact, ScalarEvaluationResult, Hook, Transmitter, Receiver, TransmitState, ReceiveState
from hermespy.core.monte_carlo import Evaluation, GridDimension
from hermespy.modem import (
    CommunicationTransmission,
    TransmittingModem,
    CommunicationWaveform,
)
from hermespy.radar import RadarReception, RadarCube
from hermespy.jcas.jcas import JCASTransmission

from compute_precoding import compute_precoding_frequency_regularized


class NewJCASTransmission(JCASTransmission):

    __base_signal: Signal

    def __init__(
        self, base_signal: Signal, transmission: CommunicationTransmission
    ) -> None:
        self.__base_signal = base_signal
        JCASTransmission.__init__(self, transmission)

    @property
    def base_signal(self) -> Signal:
        """Single-antenna stream precoded by the operator."""

        return self.__base_signal

    def to_HDF(self, group: Group) -> None:
        # Serialize base signal
        self.base_signal.to_HDF(self._create_group(group, "base_signal"))

        # Serialize base class
        JCASTransmission.to_HDF(self, group)

    @classmethod
    def from_HDF(cls, group: Group) -> NewJCASTransmission:
        # Deserialize base signal
        base_signal = Signal.from_HDF(group["base_signal"])

        # Deserialize base class
        transmission = JCASTransmission.from_HDF(group)

        return cls(base_signal, transmission)


class NewJCASReception(RadarReception):
    __residual_signal: Signal
    __corrected_signal: Signal
    __beamformed_signal: Signal

    def __init__(
        self,
        signal: Signal,
        residual_signal: Signal,
        corrected_signal: Signal,
        beamformed_signal: Signal,
        cube: RadarCube,
    ) -> None:
        RadarReception.__init__(self, signal, cube)
        self.__residual_signal = residual_signal
        self.__corrected_signal = corrected_signal
        self.__beamformed_signal = beamformed_signal

    @property
    def residual_signal(self) -> Signal:
        """Predicted signal leaking into the receiving ADCs."""

        return self.__residual_signal

    @property
    def corrected_signal(self) -> Signal:
        """Signal after digital self-interference removal."""

        return self.__corrected_signal

    @property
    def beamformed_signal(self) -> Signal:
        """Coherently combined signal across all receiving antennas."""

        return self.__beamformed_signal

    def to_HDF(self, group: Group) -> None:
        # Serialize cube
        RadarReception.to_HDF(self, group)

        # Serialize additional signals
        self.residual_signal.to_HDF(self._create_group(group, "residual_signal"))
        self.__corrected_signal.to_HDF(self._create_group(group, "corrected_signal"))
        self.__beamformed_signal.to_HDF(self._create_group(group, "beamformed_signal"))

    @classmethod
    def from_HDF(cls, group: Group) -> NewJCASReception:
        # Deserialize base class
        radar_reception = RadarReception.from_HDF(group)

        # Deserialize additional signals
        residual_signal = Signal.from_HDF(group["residual_signal"])
        corrected_signal = Signal.from_HDF(group["corrected_signal"])
        beamformed_signal = Signal.from_HDF(group["beamformed_signal"])

        return cls(
            radar_reception.signal,
            residual_signal,
            corrected_signal,
            beamformed_signal,
            radar_reception.cube,
        )


class NewJCAS(Transmitter[NewJCASTransmission], Receiver[NewJCASReception]):
    """Joint Communication and Sensing Operator.

    A combination of communication and sensing operations.
    Senses the enviroment via a correlatiom-based time of flight estimation of transmitted waveforms.
    """

    # The specific required sampling rate
    __max_range: float  # Maximally detectable range
    __waveform: CommunicationWaveform
    __transmitting_modem: TransmittingModem
    __interference_estimate: np.ndarray
    __precoding: np.ndarray
    __normalize: Literal["power", "energy", "amplitude"] | None
    __transmission: NewJCASTransmission | None

    def __init__(
        self,
        max_range: float,
        waveform: CommunicationWaveform,
        beam_focus: BeamFocus | None = None,
        normalize: Literal["power", "energy", "amplitude"] | None = None,
        reference: Device | None = None,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        seed: int | None = None,
        regularization: float = 0.0,
    ) -> None:
        """
        Args:

            max_range (float):
                Maximally detectable range in m.
        """

        # Initialize base classes
        Transmitter.__init__(self, seed, selected_transmit_ports)
        Receiver.__init__(self, seed, reference, selected_receive_ports)
        #TransmitBeamformer.__init__(self)
        #ReceiveBeamformer.__init__(self)

        # Initialize class attributes
        self.beam_focus = (
            SphericalFocus(np.zeros(2)) if beam_focus is None else beam_focus
        )
        self.__normalize = normalize
        self.__transmitting_modem = TransmittingModem()
        self.__transmitting_modem.random_mother = self
        self.__regularization = regularization
        self.__sampling_rate = None
        self.__interference_estimate = np.zeros((1, 1, 1), dtype=np.complex128)
        self.__interference_estimate_freq = np.zeros((1, 1, 1), dtype=np.complex128)
        self.__precoding = np.ones((1, 0), dtype=np.complex128)
        self.max_range = max_range
        self.waveform = waveform
        self.__transmission = None

        # Register internal transmit callback.
        # This is required in order to playback recorded drops from savefiles.
        self.add_transmit_callback(self.__cache_transmission)

    @property
    def beam_focus(self) -> BeamFocus:
        """Focused point of the algorithm."""

        return self.__beam_focus

    @beam_focus.setter
    def beam_focus(self, value: BeamFocus) -> None:
        self.__beam_focus = value

    @property
    def interference_estimate(self) -> np.ndarray:
        """Interference estimate impulse response.

        A three-dimensional numpy array of shape `(num_receive_ports, num_transmit_ports, num_samples)`.

        Note that the the internal impulse response will be truncated / padded to match
        the number of samples of the internal communication waveform.
        """

        return self.__interference_estimate

    @interference_estimate.setter
    def interference_estimate(self, value: np.ndarray) -> None:
        # Save the interference estimate
        self.__interference_estimate = value

        # Compute the frequency response of the leakage
        self.__interference_estimate_freq = fft(value, n=self.num_samples, axis=2)

    @property
    def precoding(self) -> np.ndarray:
        """Precoding applied to waveform frequency bins during transmission.

        A two-dimensional numpy array of shape `(num_transmit_ports, num_samples)`.
        """

        return self.__precoding

    @property
    def waveform(self) -> CommunicationWaveform:
        return self.__waveform

    @waveform.setter
    def waveform(self, value: CommunicationWaveform) -> None:
        self.__waveform = value
        self.__transmitting_modem.waveform = value

    @property
    def power(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.power

    def __precode(self, stream_samples: np.ndarray, state: TransmitState) -> np.ndarray:
        """Precode a given stream of samples for transmission.
        
        Args:
        
            stream_samples: Complex-valued samples to be precoded.
            state: Device for which to precode the samples.
            
        Return: Precoded samples.
        """

        # Derive a precoding matrix for self-interference cancellation
        # If no precoding is cached, compute one
        if self.__precoding.shape[1] != state.num_digital_transmit_ports:
            precoding = compute_precoding_frequency_regularized(
                self.__interference_estimate_freq,
                state.antennas.spherical_phase_response(
                    state.carrier_frequency,
                    *self.beam_focus.spherical_angles(state),
                    mode=AntennaMode.TX,
                ).conj(),
                self.__regularization,
            )
            self.__precoding = precoding
        # Otherwise, refer to the cached precoding
        else:
            precoding = self.__precoding

        input_freq = fft(stream_samples, n=precoding.shape[1])
        precoded_freq = np.empty(precoding.shape, dtype=np.complex_)
        for f, filter in enumerate(precoding):
            precoded_freq[f, :] = input_freq * filter

        precoded_samples: np.ndarray = ifft(precoded_freq, n=self.num_samples, axis=1)

        # Apply selected normalization routine
        correction_factor = 1.0
        if self.__normalize == "amplitude":
            correction_factor = np.max(
                [np.abs(precoded_samples.real), np.abs(precoded_samples.imag)]
            )

        elif self.__normalize == "power":
            correction_factor = np.mean(np.abs(precoded_samples) ** 2) ** 0.5

        elif self.__normalize == "energy":
            correction_factor = np.sum(np.abs(precoded_samples) ** 2)

        precoded_samples /= correction_factor
        return precoded_samples
    
    def __cache_transmission(self, transmission: NewJCASTransmission) -> None:
        """Callback caching a previous transmission.

        Args:

            transmission (NewJCASTransmission): The transmission to be cached.
        """

        self.__transmission = transmission

    def _transmit(self, state: TransmitState, duration: float) -> NewJCASTransmission:
        # Generate communication transmission
        communication_transmission = self.__transmitting_modem._transmit(state, duration)
        communication_signal = communication_transmission.signal
        communication_signal.carrier_frequency = state.carrier_frequency if self.carrier_frequency is None else state.carrier_frequency

        communication_samples = communication_signal.getitem(0, False)
        num_padding = self.num_samples - communication_samples.shape[0]
        if num_padding > 0:
            communication_samples = np.append(
                communication_samples, np.zeros(num_padding)
            )

        # Precode for self-interference cancellation
        precoded_signal = communication_signal.from_ndarray(
            self.__precode(communication_samples, state)
        )
        precoded_transmission = CommunicationTransmission(
            precoded_signal, communication_transmission.frames
        )

        transmission = NewJCASTransmission(
            communication_signal, precoded_transmission
        )
        return transmission

    def _receive(self, signal: Signal, state: ReceiveState) -> NewJCASReception:
        # There must be a recent transmission being cached in order to correlate
        if self.__transmission is None:
            raise RuntimeError("Reception must be preceeded by a transmission")

        precoded_freq = fft(
            self.__transmission.signal.getitem(), n=self.precoding.shape[1], axis=1
        )

        # Initialize the input signal
        input_time = signal.getitem(slice(None, None)).copy()[
            :, : self.num_required_samples
        ]
        input_time = np.append(
            input_time,
            np.zeros(
                (
                    state.num_digital_receive_ports,
                    max(0, self.num_required_samples - signal.num_samples),
                )
            ),
            axis=1,
        )

        # Predict the residual self-interference
        residual_freq = np.einsum(
            "mnl,nl->ml", self.__interference_estimate_freq, precoded_freq
        )
        residual_time = ifft(residual_freq, n=self.num_samples, axis=1)

        # Correct the input signal
        input_clean_time = input_time.copy()
        input_clean_time[:, : self.num_samples] -= residual_time

        # Coherent combination of the receive antennas
        # Boils down to a receive beamformer
        phase_response = state.antennas.spherical_phase_response(
            state.carrier_frequency,
            *self.beam_focus.spherical_angles(state),
            mode=AntennaMode.RX,
        ).conj()

        # Quick hack: Use the analog signal for range estimation
        #beamformed_time = (phase_response @ input_clean_time) / self.num_receive_ports
        beamformed_time = (phase_response @ input_time) / state.num_digital_receive_ports

        range_estimation = (
            np.abs(
                correlate(
                    beamformed_time,
                    self.__transmission.base_signal.getitem(0, None),
                    mode="valid",
                )
            )
            ** 2
            / self.num_samples
        )

        velocity_bins = np.zeros((1))
        range_bins = (
            0.5
            * correlation_lags(
                beamformed_time.size,
                self.__transmission.base_signal.num_samples,
                mode="valid",
            )
            * self.range_resolution
        )

        angle_bins = np.zeros((1, 2))

        cube_data = range_estimation[np.newaxis, np.newaxis, :]
        cube = RadarCube(
            cube_data, angle_bins, velocity_bins, range_bins, state.carrier_frequency
        )
        beamformed_signal = signal.from_ndarray(beamformed_time)

        return NewJCASReception(
            signal,
            signal.from_ndarray(residual_time),
            signal.from_ndarray(input_clean_time),
            beamformed_signal,
            cube,
        )

    @property
    def sampling_rate(self) -> float:
        modem_sampling_rate = self.waveform.sampling_rate

        if self.__sampling_rate is None:
            return modem_sampling_rate

        return max(modem_sampling_rate, self.__sampling_rate)

    @sampling_rate.setter
    def sampling_rate(self, value: float | None) -> None:
        if value is None:
            self.__sampling_rate = None
            return

        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def range_resolution(self) -> float:
        """Resolution of the Range Estimation.

        Returns:
            float:
                Resolution in m.

        Raises:

            ValueError:
                If the range resolution is smaller or equal to zero.
        """

        return speed_of_light / self.sampling_rate

    @range_resolution.setter
    def range_resolution(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Range resolution must be greater than zero")

        self.sampling_rate = speed_of_light / value

    @property
    def frame_duration(self) -> float:
        return self.waveform.frame_duration

    @property
    def max_range(self) -> float:
        """Maximally Estimated Range.

        Returns:
            The maximum range in m.

        Raises:

            ValueError:
                If `max_range` is smaller or equal to zero.
        """

        return self.__max_range

    @max_range.setter
    def max_range(self, value) -> None:
        if value <= 0.0:
            raise ValueError("Maximum range must be greater than zero")

        self.__max_range = value

    @property
    def num_samples(self) -> int:
        """Number of samples per communication frame."""

        return self.waveform.samples_per_frame

    @property
    def num_required_samples(self) -> int:
        """Number of received samples required to achieve the desired range resolution."""

        return self.waveform.samples_per_frame + 2 * ceil(
            self.max_range / self.range_resolution
        )

    def _recall_reception(self, group: Group) -> NewJCASReception:
        return NewJCASReception.from_HDF(group)

    def _recall_transmission(self, group: Group) -> NewJCASTransmission:
        return NewJCASTransmission.from_HDF(group)


class LeakingPowerArtifact(Artifact):
    """Artifact of the power leaking into a NewJCAS operator during each processing step."""

    __power: float

    def __init__(self, power: float) -> None:
        self.__power = power

    @property
    def power(self) -> float:
        return self.__power

    def __str__(self) -> str:
        return f"{self.power:.2E}"

    def to_scalar(self) -> float:
        return self.power


class LeakingPowerEvaluation(Evaluation):
    """Evaluation of the power leaking into a NewJCAS operator during each processing step."""

    __power: float

    def __init__(self, power: float) -> None:

        # Initialize base class
        Evaluation.__init__(self)

        # Initialize attributes
        self.__power = power

    @property
    def power(self) -> float:
        return self.__power

    def artifact(self) -> Artifact:
        return LeakingPowerArtifact(self.power)

    def _prepare_visualization(self, figure, axes, **kwargs):
        raise NotImplementedError("Visualization not supported")

    def _update_visualization(self, visualization, **kwargs):
        raise NotImplementedError("Visualization not supported")


class LeakingPowerEvaluator(Evaluator):
    """Evaluator for the power leaking into a NewJCAS operator during each processing step."""

    __receive_hook: Hook[NewJCASReception]
    __reception: NewJCASReception | None
    __stage: Literal['analog', 'digital']

    def __init__(self, dsp: NewJCAS, stage: Literal['analog', 'digital']) -> None:
        """ 
        
        Args:
        
            dsp: Joint communication and sensing DSP layer.
            stage: At wich point should the leaking power be considered?
        """

        # Initialize base class
        Evaluator.__init__(self)

        # Initialize attributes
        self.__stage = stage
        
        # Register callback
        self.__receive_hook = dsp.add_receive_callback(self.__receive_callback)

    def __receive_callback(self, reception: NewJCASReception) -> None:
        self.__reception = reception

    @property
    def stage(self) -> Literal['analog', 'digital']:
        return self.__stage

    @property
    def abbreviation(self) -> str:
        return "LP(A)" if self.stage == 'analog' else "LP(D)"

    @property
    def title(self) -> str:
        return f"Leaking {self.stage.title()} Power"

    def evaluate(self) -> LeakingPowerEvaluation:
        # Ensure the DSP has a reception available, i.e. has been triggered at least once
        if self.__reception is None:
            raise RuntimeError("No reception available")

        # Fetch the leaking power at the configured processing stage
        if self.stage == 'analog':
            power = np.sum(self.__reception.signal.power, keepdims=False)

        elif self.stage == 'digital':
            power = np.sum(self.__reception.corrected_signal.power, keepdims=False)

        else:
            raise RuntimeError(f"Invalid stage: {self.stage}")

        return LeakingPowerEvaluation(power)

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self, False)

    def __del__(self) -> None:
        self.__receive_hook.remove()

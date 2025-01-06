# -*- coding: utf-8 -*-

from __future__ import annotations

from hermespy.modem import RootRaisedCosineWaveform, RaisedCosineWaveform, SingleCarrierCorrelationSynchronization, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SchmidlCoxSynchronization, SchmidlCoxPilotSection, OrthogonalLeastSquaresChannelEstimation, OrthogonalZeroForcingChannelEqualization, PrefixType, GridResource, GridElement, ElementType, SymbolSection, OFDMWaveform

from parameters import *

modulation_order = 4

# Root-Raised cosine waveform configuration
rrc = RootRaisedCosineWaveform(
    symbol_rate=.25 * sampling_rate,
    oversampling_factor=24,
    num_preamble_symbols=64,
    num_data_symbols=128,
    pilot_rate=4,
    modulation_order=modulation_order,
    roll_off=.9,
)
rrc.synchronization = SingleCarrierCorrelationSynchronization()
rrc.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
rrc.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Raised cosine waveform configuration
rc = RaisedCosineWaveform(
    symbol_rate=.5 * sampling_rate,
    oversampling_factor=2,
    num_preamble_symbols=32,
    num_data_symbols=64,
    pilot_rate=4,
    modulation_order=modulation_order,
    roll_off=.9,
)
rc.synchronization = SingleCarrierCorrelationSynchronization()
rc.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
rc.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# OFDM waveform configuration
num_subcarriers = 128
num_symbols = 2
prefix_ratio = .1
prefix_type = PrefixType.CYCLIC
pilot_distance = 4

grid_resources = [
    GridResource(
        num_subcarriers // (1 + pilot_distance),
        prefix_type=prefix_type,
        prefix_ratio=prefix_ratio,
        elements=[
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, pilot_distance)
        ],
    ),
]
grid_structure = [
    SymbolSection(
        num_symbols,
        [0,],
        0,
    ),
]
ofdm = OFDMWaveform(
    grid_resources=grid_resources,
    grid_structure=grid_structure,
    subcarrier_spacing=sampling_rate / num_subcarriers,
    num_subcarriers=num_subcarriers,
    modulation_order=modulation_order,
    dc_suppression=False,
    oversampling_factor=1,
)
ofdm.pilot_section = SchmidlCoxPilotSection()
ofdm.synchronization = SchmidlCoxSynchronization()
ofdm.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
ofdm.channel_equalization = OrthogonalZeroForcingChannelEqualization()

# -*- coding: utf-8 -*-
#
# This script generates Figure 3: Calibration performance of the paper "Self-Interference Cancellation in Digital Sensing and Communication Arrays",
# to be presented at the 2025 5th IEEE International Symposium on Joint Communications & Sensing.
# It requires the results generated by the benchmark estimation script.
#
# Please refer to the README.md file for information on how to install the required dependencies.
# For additional questions, please don't hesitate to contact jan.adler@barkhauseninstitut.org .

from __future__ import annotations
import os.path as path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # type: ignore
from scipy.io import loadmat

# Set the correct matplotlib style for IEEE papers
plt.style.use(['science', 'ieee', 'no-latex'])

# Load the results from the benchmark estimation
results_path = path.join(path.dirname(__file__), '..', 'results', 'check_calibration', 'results.mat')
results = loadmat(results_path)

noise_variance = np.asarray(results['result_1']).flatten()
estimation_rmse = np.asarray(results['result_0']).flatten()

# Exclude the first value since there is no received signal for zero receive gain
noise_variance = noise_variance[1:]
estimation_rmse = estimation_rmse[1:]

fig, axes = plt.subplots(squeeze=True, tight_layout=True)
axes.semilogy(noise_variance, estimation_rmse)
axes.set_xlabel(r'$\sigma^2$')
axes.set_ylabel(r'RMSE')

plt.show()

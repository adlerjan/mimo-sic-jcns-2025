# Self-Interference Cancellation in Digital Sensing and Communication Arrays

This repostiory contains the Python source code required to reproduce the results presented in
the paper "Self-Interference Cancellation in Digital Sensing and Communication Arrays" presented at the 5th IEEE International Symposium on Joint Communications & Sensing.

It relies on the link-level evaluation package [HermesPy](https://github.com/Barkhausen-Institut/hermespy) in combination with a hardware setup consisting of two Universal Software Defined Radio Peripherals (USRPs) supporting [UHD](https://github.com/Barkhausen-Institut/usrp_uhd_wrapper).

## Installation

A clean virtual environment of Python version 3.11 is recommended,
all requirements can be downloaded and installed via
```shell
pip install -r requirements.txt
```

Please note that HermesPy binaries will be build from source and require a detectable build toolchain to be present, refer to [HermesPy's installation instructions](https://hermespy.org/installation.html#install-from-source) for further details.

## Usage

Executable entry points are files prefixed with either `benchmark_` or `plot_`.
Benchmark files control the USRPs and will record measurement datasets.
Executing plot files will post-process the measurement datasets and visualize performance graphs. Before execution, the information in `parameters.py` must be updated to reflect the current network, antenna and USRP configuration.

| Script | Description |
| ------ | ----------- |
| benchmark_estimation.py | Collects measurement data to benchmark the proposed precoding. Requires multiple executions for each candidate value of the regularization parameter lambda, with each execution involving the placement and removal of a target to be detected. |
| benchmark_precoding.py | Collects measurement data to benchmark the proposed leakage estimation algorithm. Requires an empty field of view with no target present. |
| plot_benchmark_estimation.py | Generates Figure 3: Calibration performance. |
| plot_benchmark_precoding_evm.py | Generates Figure 6: Communication performance. | 
| plot_benchmark_precoding_leakage.py | Generates Figure 4: Precoding performance. |
| plot_benchmark_precoding_power.py | Generates Figure 5: Received communication power. |
| plot_benchmark_precoding_range_power.py | Generates Figure 7: Radar range-power profile. |
| plot_benchmark_precoding_roc.py | Generates Figure 8: Detection Performance. |
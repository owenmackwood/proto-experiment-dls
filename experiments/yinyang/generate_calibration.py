import argparse
import numpy as np

import pyhalco_hicann_dls_vx_v2 as halco
import pystadls_vx_v2 as stadls
import pyhxcomm_vx as hxcomm
import calix.common
import calix.spiking.neuron

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)

args = parser.parse_args()

# # SHD
# targets = {
#         "leak": 80,
#         "reset": 80,
#         "threshold": 150,
#         "tau_mem": 10e-6,
#         "tau_syn": 10e-6,
#         "i_synin_gm": 500,
#         "membrane_capacitance": 63,
#         "refractory_time": 2.
#         }
ref = np.empty(512)
ref[:2*243] = 1e-6
ref[2*243:] = 100e-6

targets = {
        "leak": 80,
        "reset": 80,
        "threshold": 150,
        "tau_mem": 6e-6,
        "tau_syn": 6e-6,
        "i_synin_gm": 500,
        "membrane_capacitance": 63,
        "refractory_time": ref  # 100e-6  # 2e-6
        }

targets_in_weird_units = targets.copy()
targets_in_weird_units["tau_mem"] *= 1e6
targets_in_weird_units["tau_syn"] *= 1e6
targets_in_weird_units["refractory_time"] *= 1e6

with hxcomm.ManagedConnection() as connection:
    init = stadls.ExperimentInit()

    builder, _ = init.generate()
    stadls.run(connection, builder.done())

    # calibrate CADCs
    cadc_result = calix.common.cadc.calibrate(connection)

    # calibrate neurons
    neuron_result = calix.spiking.neuron.calibrate(connection, **targets_in_weird_units)

    np.savez(args.output, cadc=cadc_result, neuron=neuron_result, targets=targets)

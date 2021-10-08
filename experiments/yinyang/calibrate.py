from pathlib import Path
from typing import Dict, Union, Tuple
from numbers import Real
import numpy as np
from numpy.lib.npyio import NpzFile

"""
v_leak, v_reset: 30 to 100
v_threshold: 30 to 220

tau_mem: 0.5 us to 60 us, at maximum membrane capacitance
tau_syn: 0.3 us to 20 us

i_synin_gm: 80 to 800
membrane_capacitance: 0 to 63

refractory_time: 0.04 us to 64 us
synapse_dac_bias: 60 to 1022

If you change the membrane capacitance, we need a different leak conductivity to match the same membrane time constant. 
So if you reduce c_mem, smaller tau_mem will be feasible, but larger ones (like 60 us) will no longer work.

We don't recommend using too small capacitances unless you really need them. 
Since we have current-based synaptic inputs, their amplitude is also amplified at a lower membrane capacitance, 
but more importantly, their noise is amplified as well.
"""

ParamName = str
SetupId = str
Subpops = Dict[int, Real]
Targets = Dict[ParamName, Union[Real, Subpops]]
Key = Tuple[Tuple[ParamName, Union[Real, Tuple[int, Real]]]]
Calibration = Dict[str, np.ndarray]
SetupCalibrations = Dict[Key, Calibration]

def get_wafer_calibration(
    calibration_file: Path,
    wafer: int, fpga: int, 
    targets: Targets, 
) -> Calibration:

    calibrations = load_calibrations(calibration_file)
    wafer_calibration = calibrations[f"w{wafer}f{fpga}"]
    return wafer_calibration[targets_to_key(targets)]


def load_calibrations(calibration_file: Path) -> Dict[SetupId, SetupCalibrations]:

    if calibration_file.exists():
        c_npz: NpzFile = np.load(calibration_file, allow_pickle=True)
        calibrations = {wf: c_npz[wf].item() for wf in c_npz.files}
    else:
        print(f"No calibration file found at {calibration_file!s}")
        calibrations = {}

    return calibrations


def calibrate(
    calibration_file: Union[Path, str],
    wafer: int, fpga: int, 
    targets: Targets, 
    neurons: int, 
    prompt: bool,
    convert_to_usec: bool = False,
):

    c_file = Path(calibration_file)

    all_calibrations = load_calibrations(c_file)
    wafer_calibrations: dict = all_calibrations.setdefault(f"w{wafer}f{fpga}", {})

    wf_target_key = targets_to_key(targets)
    calibration = wafer_calibrations.setdefault(wf_target_key, {})

    if len(calibration):
        print(f"Existing calibration found: {calibration}")
    elif not prompt or input(
            f"Calibration for the chosen targets on w{wafer}f{fpga} not found. Run the calibration? [yes/NO] "
        ).lower() == "yes":
        import pystadls_vx_v2 as stadls
        import pyhxcomm_vx as hxcomm
        import calix.common
        import calix.spiking.neuron

        calib_targets = targets_to_calibrate(targets, neurons, convert_to_usec)

        with hxcomm.ManagedConnection() as connection:
            init = stadls.ExperimentInit()

            builder, _ = init.generate()
            stadls.run(connection, builder.done())

            # calibrate CADCs
            cadc_result = calix.common.cadc.calibrate(connection)

            # calibrate neurons
            neuron_result = calix.spiking.neuron.calibrate(connection, **calib_targets)

            calibration["cadc"] = cadc_result
            calibration["neuron"] = neuron_result

            np.savez(c_file, **all_calibrations)

    else:
        print("Aborting.")

    return wafer_calibrations[wf_target_key]


def targets_to_key(targets: Targets) -> Key:
    return tuple(
            (k, tuple(v.items()) if isinstance(v, dict) else v) 
            for k, v in targets.items()
        )


def targets_to_calibrate(targets: Targets, neurons: int, convert_to_us: bool):
    targets = targets.copy()
    for k, v in targets.items():
        if isinstance(v, dict):
            va = np.empty(neurons)
            bounds = tuple(v.keys())
            values = tuple(v.values())
            for b0, b1, val in zip(bounds[:-1], bounds[1:], values[:-1]):
                print(f"{k}[{b0}:{b1}] = {val:e}")
                va[b0 : b1] = val
            print(f"{k}[{bounds[-1]}:] = {values[-1]:e}")
            va[bounds[-1]:] = values[-1]
            targets[k] = va
        if np.any(targets[k] < 1e-3):
            if convert_to_us and k in ("tau_mem", "tau_syn", "refractory_time"):
                print(f"Converting {k} to useconds!")
                targets[k] *= 1e6
            else:
                raise ValueError(f"Target for {k} is too small {targets[k]} not in useconds")
    return targets


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--wafer", help="The wafer to run on.", type=int
    )
    parser.add_argument(
        "-f", "--fpga", help="The desired FPGA.", type=int
    )
    parser.add_argument(
        "-t", "--target", help="Module containg `targets` and `calibration_file`", type=str
    )
    parser.add_argument(
        "-n", "--prompt", help="Prompt before proceeding with the calibration.", 
        action="store_true", default=False,
    )
    return parser.parse_args()


args = parse_arguments()
if __name__ == "__main__":
    import importlib

    mod = importlib.import_module(args.target)

    print(f"{args.wafer=} {args.fpga=} {args.target=} {args.prompt=} {mod.calibration_file=!s}")
    calibrate(
        mod.calibration_file,
        args.wafer, args.fpga, 
        mod.targets, 
        512, args.prompt, True
    )

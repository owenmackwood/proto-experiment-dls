import sys
import numpy as np
from typing import List, Union, Dict
from pathlib import Path
import pyhxcomm_vx as hxcomm

from strobe.datasets.yinyang import YinYangDataset
from strobe.spikes import SpikeTimesToDense
from strobe.datalogger import DataLogger
from strobe.backend import LayerSize, StrobeBackend

from dlens_vx_v2 import logger
logger_fisch = logger.get("fisch")
logger.set_loglevel(logger_fisch, logger.LogLevel.ERROR)

class StrobeLayer:
    def __init__(self):
        super().__init__()
        self.on_hx = False

    def inject(
            self,
            spikes: np.ndarray = None,
            traces: np.ndarray = None,
            parameters: Dict = None,
            time_step: float = None
        ):
        self.on_hx = True

        self.spikes = spikes
        self.traces = traces
        self.time_step = time_step

        if parameters["tau_mem"] != self.params["tau_mem"]:
            raise ValueError("Neuron parameter 'tau_mem' does not match calibration target!")

        if parameters["tau_syn"] != self.params["tau_syn"]:
            raise ValueError("Neuron parameter 'tau_syn' does not match calibration target!")


class Linear(StrobeLayer):
    def __init__(self, *shape: int, scale: float) -> None:
        super().__init__()
        self.shape = shape
        self.scale = scale

class LILayer(StrobeLayer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        # self.params = params

class LIFLayer(StrobeLayer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        # self.params = params



with hxcomm.ManagedConnection() as connection:
    n_input: int = 5
    n_hidden: int = 243
    n_output: int = 3

    """
    YY constructor parameters
    """
    input_repetitions: int = 5
    n_hidden: int = 243
    readout_scaling: float = 10.0
    interpolation: int = 1

    """
    nn.Network constructor parameters
    """
    backend = None
    neuron_parameters = None

    # parameters for hardware execution
    time_step = 1.7e-6 / interpolation
    trace_offset = 0.38
    trace_scale = 1 / 0.33

    # alignment of traces and spikes from chip
    spike_shift = 1.7e-6 / interpolation
    weights = []

    record_madc = False

    """
    After constructing the nn.Network object, a call is now made to its `connect` method
    which builds up the arguments for the `StrobeBackend` constructor.
    """
    calibration: str = "cube_66.npz"
    synapse_bias: int = 1000
    sample_separation: float = 100e-6
    inference_mode: bool = False

    layer_hidden = LIFLayer(size=n_hidden)
    layer_output = LILayer(size=n_output)
    
    scale = 0.7 * (1.0 - np.exp(-1.7e-6/6e-6))
    weights_hidden = np.random.normal(
        size=(layer_hidden.size, n_input * input_repetitions), 
        loc=1e-3, 
        scale=scale / np.sqrt(n_input * input_repetitions)
        )

    weights_output = np.random.normal(
        size=(layer_output.size, layer_hidden.size), 
        scale=scale / np.sqrt(layer_hidden.size)
        )

    neuron_layers: List[Union[LILayer, LIFLayer]] = [
        layer_hidden,
        layer_output,
    ]

    structure: List[Union[int, LayerSize]] = [
        n_input * input_repetitions,
        LayerSize(layer_hidden.size, spiking=True),
        LayerSize(layer_output.size, spiking=False),
    ]

    weight_layers: List[np.ndarray] = [
        weights_hidden,
        weights_output,
    ]

    backend = StrobeBackend(
        connection, structure, calibration, synapse_bias, sample_separation
        )
    backend.configure()

    backend.load_ppu_program(Path(__file__).parent / "../../bin/strobe.bin")


seed = 0
n_samples = 25
epochs = 50

# fix seed
np.random.seed(seed)

# load data set
data_train = YinYangDataset(size=5000, seed=42)
data_test = YinYangDataset(size=1000, seed=40)

to_dense = SpikeTimesToDense(1/24/interpolation, n_samples*interpolation)



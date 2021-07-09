import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Union, Dict
from pathlib import Path
import pyhxcomm_vx as hxcomm

from strobe.datasets.yinyang import YinYangDataset
from strobe.spikes import SpikeTimesToDense
from strobe.datalogger import DataLogger
from strobe.backend import FPGA_MEMORY_SIZE, LayerSize, StrobeBackend

import torch

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

    weight_scale: float = 1000.  # 240.
    """
    YY constructor parameters
    """
    input_repetitions: int = 5
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
    calibration: str = "cube_69.npz"
    synapse_bias: int = 1000
    sample_separation: float = 5e-3
    inference_mode: bool = False

    layer_hidden = LIFLayer(size=n_hidden)
    layer_output = LILayer(size=n_output)
    
    scale = 0.7 * (1.0 - np.exp(-1.7e-6/6e-6))
    weights_hidden = np.random.normal(
        size=(n_input * input_repetitions, layer_hidden.size), 
        loc=1e-3, 
        scale=scale / np.sqrt(n_input * input_repetitions)
        )

    weights_output = np.random.normal(
        size=(layer_hidden.size, layer_output.size), 
        scale=scale / np.sqrt(layer_hidden.size)
        )
    
    print(f"Rescaling weights by {weight_scale}")
    for w in (weights_hidden, weights_output):
        w *= weight_scale

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

    # backend.load_ppu_program(Path(__file__).parent / "../../bin/strobe.bin")
    backend.load_ppu_program(f'{Path("~/workspace/bin/strobe.bin").expanduser()}')

    seed = 0
    n_samples = 25
    epochs = 50
    batch_size = 10  # 100
    n_steps = n_samples * interpolation

    # fix seed
    np.random.seed(seed)

    # load data set
    data_train = YinYangDataset(size=5000, seed=42)
    data_test = YinYangDataset(size=1000, seed=40)

    # train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(data_test, batch_size=len(data_test), shuffle=False)

    to_dense = SpikeTimesToDense(1/24/interpolation, n_samples*interpolation)

    # datalogger = DataLogger(epochs + 1, len(train_loader))

    max_hw_batch_size = int(np.floor(FPGA_MEMORY_SIZE / n_steps / backend._n_vectors / 128))
    hw_batch_size = min(batch_size, max_hw_batch_size)

    labels = np.arange(n_input) + 256
    input_spikes = []
    for b in range(batch_size):
        x, y = data_train[b]
        times = x * time_step + spike_shift
        order = np.argsort(times)
        input_spikes.append(np.vstack([times[order], labels[order]]).T)

    backend.write_weights(*weight_layers)

    # batch_durations = np.zeros((batch_size, 2))
    for trial in range(5):
        spikes, traces, durations, causal_traces = backend.run(
                input_spikes,
                n_samples=n_steps // interpolation,
                record_madc=record_madc,
                trigger_reset=inference_mode)
        # batch_durations[:] = np.array(durations)
        if not (np.array(durations) > 85200).any():
            print("Success!")
            break
        else:
            print("Took too long!")
            pass


    fig = plt.Figure()
    gs = GridSpec(3, 3)

    for t_i, trace in enumerate(causal_traces[:-1]):
        ij = np.unravel_index(t_i, shape=(3, 3))
        ax = fig.add_subplot(gs[ij])
        ax.imshow(trace, aspect="auto", interpolation="nearest", vmin=0)
    fig.savefig(Path("~/plots/traces.png").expanduser())


    # normalize membrane traces
    for t in traces:
        t -= trace_offset
        t *= trace_scale
        t -= t[:, 0, None]

    layered_traces = []
    layered_spikes = []
    for l, layer in enumerate(neuron_layers):
        layered_traces.append(np.zeros((batch_size, n_steps, layer.size)))
        layered_spikes.append(np.zeros((batch_size, n_steps, layer.size)))
    
    fig: plt.Figure() = plt.figure()
    gs = GridSpec(len(neuron_layers), 1)
    for l, layer in enumerate(neuron_layers):
        for i in range(interpolation):
            layered_traces[l][:, i::interpolation, :] = traces[l]
        ax = fig.add_subplot(gs[l])
        ax.plot(layered_traces[l][0, ...])
    fig.savefig(Path("~/plots/potentials.png").expanduser())


    for b in range(batch_size):
        for l, layer in enumerate(neuron_layers):
            print(f"Batch {b}, Layer {l}, spikes: {spikes[l][b].size}")
            if spikes[l][b].size:
                hist = np.zeros((n_steps, layer.size))
                spike_times = spikes[l][b][:, 0] - spike_shift
                mask = spike_times < time_step * n_steps
                units = spikes[l][b][:, 1].astype(np.int)
                hist[(spike_times[mask] // time_step).astype(np.int), units[mask]] = 1
                layered_spikes[l][b, :, :] = hist

    fig: plt.Figure() = plt.figure()
    gs = GridSpec(1, len(neuron_layers))
    for l, layer in enumerate(neuron_layers):
        ax = fig.add_subplot(gs[l])
        ax.imshow(layered_spikes[l][0, ...], aspect="auto", interpolation="nearest")
    fig.savefig(Path("~/plots/spikes.png").expanduser())



    # def forward(x):
    #     """
    #     This forward function is defined in strobe.nn.Network
    #     It takes a dense representation of `x` which is computed in YY.__call__
    #     before being passed to the network.
    #     """

    #     # extract number of samples from input tensor
    #     batch_size = x.shape[0]
    #     n_steps = x.shape[1]

    #     # calculate maximum batch size where traces fit into FPGA memory
    #     max_hw_batch_size = int(np.floor(FPGA_MEMORY_SIZE / n_steps / backend._n_vectors / 128))
    #     hw_batch_size = min(batch_size, max_hw_batch_size)

    #     # self.synchronize_hardware()
    #     backend.write_weights(*weight_layers)

    #     batch_durations = np.zeros((batch_size, 2))

    #     layered_traces = []
    #     layered_spikes = []
    #     for l, layer in enumerate(neuron_layers):
    #         layered_traces.append(np.zeros((batch_size, n_steps, layer.size)))
    #         layered_spikes.append(np.zeros((batch_size, n_steps, layer.size)))

    #     hw_batch_bounds = np.arange(0, batch_size, hw_batch_size)
    #     for s in [slice(i, min(batch_size + 1, i + hw_batch_size)) for i in hw_batch_bounds]:
    #         hw_x = x[s, :, :]

    #         input_spikes = []
    #         for b in range(hw_x.shape[0]):
    #             spike_bins = np.where(hw_x[b].T.cpu())
    #             labels = spike_bins[0] + 256
    #             times = spike_bins[1].astype(np.float) * time_step + spike_shift

    #             # sort spike train according to injection times
    #             order = np.argsort(times)
    #             input_spikes.append(np.vstack([times[order], labels[order]]).T)

    #         for trial in range(5):
    #             spikes, traces, durations = backend.run(
    #                     input_spikes,
    #                     n_samples=n_steps // interpolation,
    #                     record_madc=record_madc,
    #                     trigger_reset=inference_mode)
    #             batch_durations[s, :] = np.array(durations)
    #             if not (np.array(durations) > 85200).any():
    #                 break
    #             else:
    #                 pass

    #         # normalize membrane traces
    #         for t in traces:
    #             t -= trace_offset
    #             t *= trace_scale
    #             t -= t[:, 0, None]

    #         for l, layer in enumerate(neuron_layers):
    #             for i in range(interpolation):
    #                 layered_traces[l][s, i::interpolation, :] = traces[l]

    #         for b in range(hw_x.shape[0]):
    #             for l, layer in enumerate(neuron_layers):
    #                 if spikes[l][b].size:
    #                     hist = np.zeros((n_steps, layer.size))
    #                     spike_times = spikes[l][b][:, 0] - spike_shift
    #                     mask = spike_times < time_step * n_steps
    #                     units = spikes[l][b][:, 1].astype(np.int)
    #                     hist[(spike_times[mask] // time_step).astype(np.int), units[mask]] = 1
    #                     layered_spikes[l][s, :, :][b, :, :] = hist

    #     for l, layer in enumerate(neuron_layers):
    #         layer.inject(layered_spikes[l], layered_traces[l], neuron_parameters, time_step)


    # def yy_call(x, y=None):
    #     times = x.repeat(1, input_repetitions)
    #     spikes = to_dense(times)
    #     spikes = spikes.view((spikes.shape[0], spikes.shape[1], -1))

    #     output = forward(spikes)

    #     # subtract margin to get rid of noise floor
    #     margin = 0.35

    #     # get max-over-time and extract classifiation response
    #     m, _ = torch.max(output - margin, 1)
    #     _, am = torch.max(m, 1)

    #     # return prediction in case no target was given
    #     if y is None:
    #         return am
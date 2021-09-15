import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from itertools import product

from numpy.core.fromnumeric import shape
from yy import neuron_layers, interpolation, trace_offset, trace_scale
from yy import batch_size, n_steps, spike_shift, time_step
import yy

save_path = Path("~/data").expanduser()
all_inputs = np.load(save_path / "input_spikes.npz")
traces = np.load(save_path / "membranes.npz")
spikes = [np.load(save_path / f"spikes_layer{l}.npz") for l, layer in enumerate(neuron_layers)]
causal_traces = np.load(save_path / "causal_traces.npz")
computed_traces = np.load(save_path / "computed_traces.npz")

all_inputs = [all_inputs[n] for n in all_inputs.files]
traces = [traces[n] for n in traces.files]
spikes = [[s[n] for n in s.files] for s in spikes]
causal_traces = [causal_traces[n] for n in causal_traces.files]
traces_hidden = computed_traces["traces_hidden"]
traces_output = computed_traces["traces_output"]

fig_hw = plt.Figure()
gs = GridSpec(3, 3)

for t_i, trace in enumerate(causal_traces[:-1]):
    ij = np.unravel_index(t_i, shape=(3, 3))
    ax = fig_hw.add_subplot(gs[ij])
    ax.imshow(trace, aspect="auto", interpolation="nearest", vmin=0)
fig_hw.savefig(Path("~/plots/traces_hardware.png").expanduser())

# normalize membrane traces
for t in traces:
    t -= trace_offset
    t *= trace_scale
    t -= t[:, 0, None]

layered_traces = []
layered_spikes = []
for l, layer in enumerate(neuron_layers):
    layered_traces.append(np.zeros((batch_size, n_steps, layer.size)))
    layered_spikes.append(np.zeros((n_steps, layer.size, batch_size)))

fig: plt.Figure() = plt.figure()
gs = GridSpec(len(neuron_layers), 1)
for l, layer in enumerate(neuron_layers):
    for i in range(interpolation):
        layered_traces[l][:, i::interpolation, :] = traces[l]
    ax = fig.add_subplot(gs[l])
    ax.plot(layered_traces[l][..., 0])
fig.savefig(Path("~/plots/potentials.png").expanduser())

            
for b in range(batch_size):
    for l, layer in enumerate(neuron_layers):
        print(f"Batch {b}, Layer {l}, spikes: {spikes[l][b].shape[0]}")
        if spikes[l][b].size:
            hist = np.zeros((n_steps, layer.size))
            spike_times = spikes[l][b][:, 0] - spike_shift
            mask = spike_times < time_step * n_steps
            units = spikes[l][b][:, 1].astype(int)
            hist[(spike_times[mask] // time_step).astype(int), units[mask]] = 1
            layered_spikes[l][..., b] = hist

for l, layer in enumerate(neuron_layers):
    for dn, data in {"spikes":layered_spikes, "traces":[traces_hidden, traces_output]}.items():
        fig: plt.Figure() = plt.figure()
        gs = GridSpec(3, 3)
        for b in range(batch_size-1):
            ij = np.unravel_index(b, shape=(gs.nrows, gs.ncols))
            ax = fig.add_subplot(gs[ij])
            ax.imshow(data[l][..., b], aspect="auto", interpolation="nearest")
        fig.savefig(Path(f"~/plots/{dn}_{'output' if l else 'hidden'}.png").expanduser())

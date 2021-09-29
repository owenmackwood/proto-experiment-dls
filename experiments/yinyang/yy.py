import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from typing import List, Union, Dict, Tuple
from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from numba import njit
from strobe.backend import StrobeBackend

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

n_input: int = 5
n_hidden: int = 243
n_output: int = 3

"""
YY constructor parameters
"""
input_repetitions: int = 15
interpolation: int = 1

"""
nn.Network constructor parameters
"""
# parameters for hardware execution
time_step = 1.7e-6 / interpolation
trace_offset = 0.38
trace_scale = 1 / 0.33

# alignment of traces and spikes from chip
spike_shift = 1.7e-6 / interpolation

record_madc = False

"""
After constructing the nn.Network object, a call is now made to its `connect` method
which builds up the arguments for the `StrobeBackend` constructor.
"""
calibration: str = "cube_69.npz"
synapse_bias: int = 1000
sample_separation: float = 10e-3  # Refractory time is 100e-6!
inference_mode: bool = False

layer_hidden = LIFLayer(size=n_hidden)
layer_output = LILayer(size=n_output)

# weight_scale: float = 4 * 240.  # 1000.  # 
scale = 2.5 * 240. * 0.7 * (1.0 - np.exp(-1.7e-6/6e-6))

neuron_layers: List[Union[LILayer, LIFLayer]] = [
    layer_hidden,
    layer_output,
]


class LayerSize(int):
    def __new__(cls, size, recurrent=False, spiking=True):
        self = int.__new__(cls, size)
        self.recurrent = recurrent
        self.spiking = spiking
        return self


structure: List[Union[int, LayerSize]] = [
    n_input * input_repetitions,
    LayerSize(layer_hidden.size, spiking=True),
    LayerSize(layer_output.size, spiking=False),
]

seed = 2
n_samples = 25
n_steps = n_samples * interpolation

tau_stdp = 6e-6  # TODO: Change mean weights to 0, let regularizer handle spikes.
eta = 1e-1
epochs = 40
batch_size = 100
train_size = batch_size*50
test_size = batch_size*20

lr_step_size = 10
# lr_gamma = 0.92
lr_factor = 0
gamma0 = 0.1 / 63


tb_str = f"_e{epochs}_bs{batch_size}_tr_{train_size}_eta{eta:.0e}_fac{lr_factor}_step{lr_step_size}_tau{tau_stdp*1e6:.1f}us_input_{input_repetitions}x"


def main():
    import pyhxcomm_vx as hxcomm
    from functools import partial
    from strobe.datasets.yinyang import YinYangDataset
    from strobe.backend import FPGA_MEMORY_SIZE, StrobeBackend

    # import torch
    # from dlens_vx_v2 import logger
    # logger_fisch = logger.get("fisch")
    # logger.set_loglevel(logger_fisch, logger.LogLevel.DEBUG)
    # logger.default_config(level=logger.LogLevel.DEBUG)
    # logger.append_to_file("all.log")

    log_dir = Path.home()/f"tboards/{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}{tb_str}"

    with hxcomm.ManagedConnection() as connection, \
        SummaryWriter(log_dir) as tb:

        # fix seed
        np.random.seed(seed)

        weights_hidden = np.random.normal(
            size=(n_input * input_repetitions, layer_hidden.size), 
            loc=10,  # 1e-3, # 
            scale=scale / np.sqrt(n_input * input_repetitions)
            )

        weights_output = np.random.normal(
            size=(layer_hidden.size, layer_output.size), 
            loc=10.,
            scale=2*scale / np.sqrt(layer_hidden.size)
            )

        weight_layers: List[np.ndarray] = [
            weights_hidden,
            weights_output,
        ]

        backend = StrobeBackend(
            connection, structure, calibration, synapse_bias, sample_separation
            )
        backend.configure()

        # backend.load_ppu_program(Path(__file__).parent / "../../bin/strobe.bin")
        backend.load_ppu_program(f'{Path.home()/"workspace/bin/strobe.bin"}')

        # load data set
        r_big = 0.5
        r_small = 0.1
        data_train = YinYangDataset(r_small, r_big, size=train_size, seed=seed)
        data_test = YinYangDataset(r_small, r_big, size=test_size, seed=seed+1)

        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=len(data_test), shuffle=False)

        max_hw_batch_size = int(np.floor(FPGA_MEMORY_SIZE / n_steps / backend._n_vectors / 128))
        print(f"Max batch size: {max_hw_batch_size}")

        m_output = np.zeros_like(weights_output)
        v_output = np.zeros_like(weights_output)
        m_hidden = np.zeros_like(weights_hidden)
        v_hidden = np.zeros_like(weights_hidden)

        forward_p = partial(forward, backend, tb, weight_layers, m_output, v_output, m_hidden, v_hidden, max_hw_batch_size)

        for epoch in range(epochs):
            print(80*"=")
            print(f"Epoch {epoch+1} of {epochs}")
            t_start_e = time.time()

            t_backend_train, t_traces, t_weight_update = forward_p(epoch, train_loader, True)
            print(20*"=", "Testing", 20*"=")
            t_backend_test, _, _ = forward_p(epoch, test_loader, False)

            t_epoch = time.time() - t_start_e
            backend_str = f"(Backend: {t_backend_train:.1f} sec training, {t_backend_test:.1f} sec testing)"
            print(f"Time {t_epoch:.1f} sec {backend_str} Traces: {t_traces:.3f}, Weight updates: {t_weight_update:.3f} sec")

        # save_path = Path("~/data").expanduser()
        # save_path.mkdir(exist_ok=True)
        # np.savez(save_path / "input_spikes", *all_inputs)
        # np.savez(save_path / "membranes", *traces)
        # for l, layer in enumerate(neuron_layers):
        #     np.savez(save_path / f"spikes_layer{l}", *spikes[l])
        # np.savez(save_path / "causal_traces", *causal_traces)
        # np.savez(save_path / "computed_traces", traces_hidden=traces_hidden, traces_output=traces_output)


def forward(
        backend: StrobeBackend,
        tb: SummaryWriter,
        weight_layers: List[np.ndarray],
        m_output: np.ndarray,
        v_output: np.ndarray,
        m_hidden: np.ndarray,
        v_hidden: np.ndarray,
        max_hw_batch_size: int, 
        epoch: int, 
        data_loader: torch.utils.data.DataLoader, 
        update_weights: bool,
    ) -> Tuple[float, float, float]:

    t_backend = 0.
    t_weight_update = 0.
    t_traces = 0.

    dataset_size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    num_batches = dataset_size // batch_size
    hw_batch_size = min(batch_size, max_hw_batch_size)
    hw_batch_bounds = np.arange(0, batch_size, hw_batch_size)

    weights_hidden, weights_output = weight_layers
    np.clip(weights_hidden, -63, 63, out=weights_hidden)
    np.clip(weights_output, -63, 63, out=weights_output)

    weight_bins = np.linspace(-63.001, 63.001, 10*63)
    prob_bins = np.linspace(0., 1., 40)
    tau_bins = np.linspace(0., 5e6 * (time_step + spike_shift), 40)

    in_ds = np.zeros((dataset_size, n_input))
    y_hat_ds = np.zeros((dataset_size, n_output))
    tau_k = np.zeros(n_output)

    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):

        input_spikes = []
        hidden_spikes = []

        batch_slice = slice(batch_idx*batch_size, (batch_idx+1)*batch_size)
        in_all = in_ds[batch_slice, :]
        y_hat = y_hat_ds[batch_slice, :]
        y_all = np.zeros((batch_size, n_output))
        c_all = np.zeros(batch_size, dtype=int)
        tau_all = np.zeros((batch_size, n_output))

        traces_hidden = np.zeros((batch_size, n_input, n_hidden))
        traces_output = np.zeros((batch_size, n_hidden, n_output))
        spikes_per_hidden = np.zeros((batch_size, n_hidden))
        spikes_per_output = np.zeros((batch_size, n_output))

        labels = np.arange(n_input) + 256
        for b in range(batch_size):
            x = batch_x[b, :]
            y = batch_y[b]
            # x, y = data_train[b + epoch*b]
            c_all[b] = y
            y_all[b, y] = 1
            times = x * time_step + spike_shift
            order = np.argsort(times)
            input_spikes.append(np.vstack([times[order], labels[order]]).T)
            in_all[b, :] = times

        backend.write_weights(*weight_layers)

        for s in [slice(i, min(batch_size + 1, i + hw_batch_size)) for i in hw_batch_bounds]:
            # batch_durations = np.zeros((batch_size, 2))
            t_start_b = time.time()
            for trial in range(5):
                spikes, traces, durations, causal_traces = backend.run(
                        input_spikes[s.start:s.stop],
                        n_samples=n_steps // interpolation,
                        record_madc=record_madc,
                        trigger_reset=inference_mode)
                # batch_durations[:] = np.array(durations)
                if not (np.array(durations) > 85200).any():
                    # print("Success!")
                    break
                else:
                    print(f"Took too long! {np.max(durations)}")
                    pass
            t_backend += time.time() - t_start_b

            if update_weights:
                t_start_trace = time.time()
                compute_traces(
                        in_all[s.start:s.stop, :], spikes, hw_batch_size, traces_hidden[s, ...], traces_output[s, ...]
                    )
                t_traces += time.time() - t_start_trace

            no_outputs = 0
            min_hidden = 10*3
            for b in range(s.stop - s.start):
                spikes_output = spikes[1][b][:, 0] - spike_shift
                spikes_hidden = spikes[0][b][:, 0] - spike_shift
                units_output = spikes[1][b][:, 1].astype(int)
                units_hidden = spikes[0][b][:, 1].astype(int)
                np.add.at(spikes_per_output[s.start+b, :], units_output, 1)
                np.add.at(spikes_per_hidden[s.start+b, :], units_hidden, 1)

                tau_k[:] = np.infty
                tau_k[units_output] = spikes_output
                no_outputs += spikes_output.size == 0
                min_hidden = min(min_hidden, units_hidden.size)
                for o in range(n_output):
                    if (units_output == o).sum() > 1:
                        print(f"Output unit {o} produced more than one spike! {units_output} {spikes_output}")
                        tau_k[o] = np.min(spikes_output[units_output == o])
                
                tau_k *= 1e6  # Bring usec values into .1-1 sec range
                
                nu = 1. + 9.*(epoch / epochs)  # Softmax sharpness
                y_hat[s.start+b, :] = activation_tau(tau_k, nu)

                tau_all[s.start+b, :] = tau_k
                hidden_spikes.append(1e6*spikes_hidden)

        err = y_hat - y_all  # shape=(batch_size, n_output)
        class_estimate = np.argmax(y_hat, axis=1)
        n_correct = (c_all == class_estimate).sum()
        accuracy = n_correct / batch_size

        # cosine_similarity = np.einsum("bi,bi->b", y_hat, y_all)

        # Add small constant to y_hat to avoid numerical problems in computing the cross entropy.
        cross_entropy = -np.sum(y_all * np.log(y_hat + 1e-32), axis=1)
        mean_loss = cross_entropy.mean()
        if no_outputs:
            print(f"No output spikes: {no_outputs}, min hidden: {min_hidden}")
        print(f"Batch: {batch_idx+1}/{num_batches}, Accuracy: {accuracy:.2f}, Loss: {mean_loss:.3f}")  # ", CS: {[f'{v:.3e}' for v in cs]}")

        if update_weights:
            t_start_w = time.time()
            # weights_rounded = np.round(weights_output)
            dw_out = (traces_output * err[:, None, :]).sum(axis=0)  # (batch_size, n_hidden, n_output) * (batch_size, n_output) -> (n_hidden, n_output)
            wt = (weights_output/63)[None, ...] * traces_output  # shape=(batch_size, n_hidden, n_output)
            bpe = np.einsum("bij,bj->bi", wt, err)  # (batch_size, n_hidden, n_output) (batch_size, n_output) -> (batch_size, n_hidden)
            dw_hidden = (traces_hidden * bpe[:, None, :]).sum(axis=0)  # (batch_size, n_input, n_hidden) (batch_size, n_hidden) -> (n_input, n_hidden)
            dw_hidden = np.vstack(input_repetitions*[dw_hidden])
            dw_out /= batch_size
            dw_hidden /= batch_size

            def regularizer(weights, spikes, target):
                sp = spikes.sum(axis=0) >= target
                sp = np.vstack(weights.shape[0] * [sp])
                return np.where(sp, 0, -gamma0 * np.abs(weights))
            
            r0_hidden = regularizer(weights_hidden, spikes_per_hidden, batch_size / n_output / 2)
            r0_output = regularizer(weights_output, spikes_per_output, batch_size / n_output)
            mean_r0_hidden = r0_hidden.mean()
            mean_r0_output = r0_output.mean()
            dw_hidden += r0_hidden
            dw_out += r0_output

            use_r1_reg = False
            if use_r1_reg:
                lambda0 = 0.01 / 63
                r1_hidden = lambda0 * weights_hidden
                r1_output = lambda0 * weights_output
                dw_hidden += r1_hidden
                dw_out += r1_output
                mean_r1_hidden = r1_hidden.mean()
                mean_r1_output = r1_output.mean()
        
            adam_update(eta, weights_hidden, m_hidden, v_hidden, dw_hidden, epoch)
            eta_hat = adam_update(eta, weights_output, m_output, v_output, dw_out, epoch)

            t_weight_update += time.time() - t_start_w

            assert np.all(np.isfinite(weights_hidden)), "Non-finite hidden weights"
            assert np.all(np.isfinite(weights_output)), "Non-finite output weights"
            np.clip(weights_hidden, -63, 63, out=weights_hidden)
            np.clip(weights_output, -63, 63, out=weights_output)

            tb_i = epoch * num_batches + batch_idx
            # fig = plt.figure()
            # gs = GridSpec(1, 1, left=0, right=1, bottom=0, top=1)
            # ax = fig.add_subplot(gs[0])
            # ax.set(xticks=(), yticks=())
            for cls in range(n_output):
                cls_at = c_all == cls
                cls_accuracy = (class_estimate[cls_at] == cls).sum() / cls_at.sum()
                tb.add_scalar(f"accuracy/class_{data_loader.dataset.class_names[cls]}", cls_accuracy, tb_i)
            #     input_cls = in_all[class_estimate == cls, :]
            #     ax.scatter(input_cls[:, 0], input_cls[:, 1], color=("r", "g", "b")[cls])
            # tb.add_figure(f"train_images{epoch}", fig, batch_idx)
            tb.add_scalar("accuracy/combined", accuracy, tb_i)
            tb.add_scalar("Loss", mean_loss, tb_i)

            tb.add_histogram("class/probability", y_hat, tb_i, bins=prob_bins)
            tb.add_histogram("class/cross_entropy", cross_entropy, tb_i)
            tb.add_histogram("class_id/estimate", class_estimate, tb_i)
            tb.add_histogram("class_id/true", c_all, tb_i)
            tb.add_histogram("hidden/spike_latency", np.hstack(hidden_spikes), tb_i, bins=tau_bins)
            tb.add_histogram("output/spike_latency", np.where(np.isfinite(tau_all), tau_all, tau_bins[-1]), tb_i, bins=tau_bins)
            tb.add_histogram("hidden/spike_counts", spikes_per_hidden, tb_i)
            tb.add_histogram("output/spike_counts", spikes_per_output, tb_i)
            tb.add_histogram("hidden/weights", weights_hidden, tb_i, bins=weight_bins)
            tb.add_histogram("output/weights", weights_output, tb_i, bins=weight_bins)

            tb.add_scalar("Learning rate", eta_hat, tb_i)
            tb.add_scalar("reg/spikes_hidden", mean_r0_hidden, tb_i)
            tb.add_scalar("reg/spikes_output", mean_r0_output, tb_i)
            tb.add_histogram("hidden/traces", traces_hidden, tb_i)
            tb.add_histogram("output/traces", traces_output, tb_i)
            tb.add_histogram("hidden/grad", dw_hidden, tb_i)
            tb.add_histogram("output/grad", dw_out, tb_i)
            tb.add_histogram("regularization/spikes_hidden", r0_hidden, tb_i)
            tb.add_histogram("regularization/spikes_output", r0_output, tb_i)
            if use_r1_reg:
                tb.add_scalar("reg/weights_hidden", mean_r1_hidden, tb_i)
                tb.add_scalar("reg/weights_output", mean_r1_output, tb_i)
                tb.add_histogram("regularization/weights_hidden", r1_hidden, tb_i)
                tb.add_histogram("regularization/weights_output", r1_output, tb_i)
        else:
            for cls in range(n_output):
                cls_at = c_all == cls
                cls_accuracy = (class_estimate[cls_at] == cls).sum() / cls_at.sum()
                tb.add_scalar(f"test/class_{data_loader.dataset.class_names[cls]}", cls_accuracy, epoch)
            
            tb.add_scalar("test/combined", accuracy, epoch)
            tb.add_scalar("test/Loss", mean_loss, epoch)

    if not update_weights:
        fig = plt.figure()
        gs = GridSpec(1, 1, left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(gs[0])
        ax.set(xticks=(), yticks=())
        fig_t = plt.figure(figsize=(10, 3))
        gs = GridSpec(1, 3, left=0, right=1, bottom=0, top=1, wspace=0)

        class_estimate_ds = np.argmax(y_hat_ds, axis=1)
        for cls in range(n_output):
            input_cls = in_ds[class_estimate_ds == cls, :]
            ax.scatter(input_cls[:, 0], input_cls[:, 1], color=("r", "g", "b")[cls])

            ax_cls = fig_t.add_subplot(gs[cls])
            ax_cls.set(xticks=(), yticks=())
            ax_cls.scatter(in_ds[:, 0], in_ds[:, 1], color=cm.viridis(y_hat_ds[:, cls]))
        
        tb.add_figure("test_classes", fig_t, epoch)
        tb.add_figure("test_images", fig, epoch)
    return t_backend, t_traces, t_weight_update


def adam_update(eta, w, m, v, dw, epoch):
    adam_eps = 1e-8
    adam_beta1 = 0.9   # .1  # .5  # 
    adam_beta2 = 0.999  # .1999 # .9  # 

    # epoch = int(t / dt_update_weights)
    # lr_epoch = epoch // lr_step_size
    # eta_hat = eta * lr_gamma**lr_epoch
    # print(f"Epoch: {epoch}, LR-Epoch: {lr_epoch}, eta_hat: {eta_hat:.2e}")
    lr_epoch = (epoch+1) // lr_step_size
    eta_hat = eta * 10**(-lr_factor * lr_epoch * (lr_step_size / epochs))
    m *= adam_beta1
    m += (1 - adam_beta1) * dw
    v *= adam_beta2
    v += (1 - adam_beta2) * np.power(dw, 2)
    m_hat = m / (1 - adam_beta1)
    v_hat = v / (1 - adam_beta2)
    w -= eta_hat * m_hat / (np.sqrt(v_hat) + adam_eps)
    return eta_hat


def activation_tau(tau_k, nu):
    """
    tau_k should be normalized such that its values are order-1.
    """

    if np.any(np.isfinite(tau_k)):
        exp_nu_tau = np.exp(-nu * tau_k)
    else:
        exp_nu_tau = np.ones_like(tau_k)
    return exp_nu_tau / np.sum(exp_nu_tau)


@njit(cache=True)
def compute_trace(
        spikes_pre, spikes_post, units_pre, units_post, trace
):
    """Assumes spikes are sorted from earliest-to-latest"""

    for j, t_post in zip(units_post, spikes_post):
        for i, t_pre in zip(units_pre, spikes_pre):
            delta_t = t_post - t_pre
            if delta_t < 0.:
                break
            trace[i, j] += np.exp(-delta_t/tau_stdp)

    # from itertools import product
    # for i, j in product(range(n_pre), range(n_post)):
    #     spi = np.argwhere(units_pre == i)
    #     spj = np.argwhere(units_post == j)
    #     if spi.size and spj.size:
    #         sti = spikes_pre[spi]
    #         stj = spikes_post[spj]
    #         for t_post in stj:
    #             t_pre = sti[sti < t_post]
    #             trace[i, j] += np.exp(-(t_post - t_pre)/tau_stdp).sum()


def compute_traces(
    all_inputs, spikes, batch_size, traces_hidden, traces_output
):
    all_inputs = all_inputs.copy()
    for b in range(batch_size):
        units_input = np.arange(n_input)
        spike_times_input = all_inputs[b, :]
        spike_times_hidden = spikes[0][b][:, 0] - spike_shift
        units_hidden = spikes[0][b][:, 1].astype(int)
        spike_times_output = spikes[1][b][:, 0] - spike_shift
        units_output = spikes[1][b][:, 1].astype(int)

        for units, times in (
                (units_input, spike_times_input),
                (units_hidden, spike_times_hidden), 
                (units_output, spike_times_output),
            ):
            idx_order = np.argsort(times)
            times[:] = times[idx_order]
            units[:] = units[idx_order]

        compute_trace(
            spike_times_input, spike_times_hidden,
            units_input, units_hidden,
            traces_hidden[b, ...]
        )
        compute_trace(
            spike_times_hidden, spike_times_output,
            units_hidden, units_output,
            traces_output[b, ...]
        )
        # print(f"{b} Hidden trace max: {traces_hidden.max()} non-zero: {np.sum(traces_hidden>0)}, spikes {units_hidden.size}")
        # print(f"{b} Output trace max: {traces_output.max()} non-zero: {np.sum(traces_output>0)}, spikes {units_output.size}")
    return traces_hidden, traces_output


if __name__ == "__main__":
    main()
    # for _ in range(20):
    #     try:
    #         main()
    #     except RuntimeError as e:
    #         print(f"Failed with {e}, trying again.")
    #         time.sleep(10.)
    #     else:
    #         break

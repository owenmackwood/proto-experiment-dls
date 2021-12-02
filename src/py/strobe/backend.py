import enum
import warnings
import numpy as np

import pyhxcomm_vx as hxcomm
import pyhaldls_vx_v2 as haldls
import pystadls_vx_v2 as stadls
import pyfisch_vx as fisch
import pylola_vx_v2 as lola
import pyhalco_hicann_dls_vx_v2 as halco
import calix.common
import calix.spiking
import gonzales

from .routing import RoutingGenerator


class PPUSignal(enum.Enum):
    RUN = 0
    NONE = 1
    HALT = 2
    RESET_BATCH = 3
    RUN_AND_RESET = 4


class LayerSize(int):
    def __new__(cls, size, recurrent=False, spiking=True):
        self = int.__new__(cls, size)
        self.recurrent = recurrent
        self.spiking = spiking
        return self


active_crossbar_node = haldls.CrossbarNode()
active_crossbar_node.mask = 0
active_crossbar_node.target = 0
enable_recurrency_builder = stadls.PlaybackProgramBuilder()
for i in range(8):
    enable_recurrency_builder.write(
        halco.CrossbarNodeOnDLS(
            halco.CrossbarOutputOnDLS(i % 4),
            halco.CrossbarInputOnDLS(i)
        ), active_crossbar_node)
    enable_recurrency_builder.write(
        halco.CrossbarNodeOnDLS(
            halco.CrossbarOutputOnDLS(4 + (i % 4)),
            halco.CrossbarInputOnDLS(i)
        ), active_crossbar_node)

silent_crossbar_node = haldls.CrossbarNode()
silent_crossbar_node.mask = 0
silent_crossbar_node.target = 2**14 - 1
disable_recurrency_builder = stadls.PlaybackProgramBuilder()
for i in range(8):
    disable_recurrency_builder.write(
            halco.CrossbarNodeOnDLS(
                halco.CrossbarOutputOnDLS(i % 4),
                halco.CrossbarInputOnDLS(i)
                ), silent_crossbar_node)
    disable_recurrency_builder.write(
            halco.CrossbarNodeOnDLS(
                halco.CrossbarOutputOnDLS(4 + (i % 4)),
                halco.CrossbarInputOnDLS(i)
                ), silent_crossbar_node)


FPGA_MEMORY_SIZE = 131072  # bytes


class StrobeBackend:
    def __init__(self, connection, structure=[256, 118, 10], calibration=None, synapse_bias=1000, sample_separation=500e-6, measure_correlation=False):
        self._connection = connection
        self.structure = structure

        self.synapse_bias = synapse_bias
        self.sample_separation = sample_separation
        self._measure_correlation = measure_correlation

        # check if first layer is recurrent
        first_recurrent = isinstance(self.structure[1], LayerSize) and self.structure[1].recurrent
        if first_recurrent:
            self._input_shift = self.structure[1]
            print(f"Shifting input by {self.structure[1]}")
        else:
            self._input_shift = 0

        self.max_hw_batch_size = 40

        # calib = np.load(calibration, allow_pickle=True)
        self._cadc_calib = calibration["cadc"]
        self._neuron_calib = calibration["neuron"]

        # this is kinda fixed for the model, we keep it here only for the sake of completeness
        self._neuron_size = 2
        self._signed_synapses = True

        # number of vectors of 128 neurons each that have to be recorded to cover the whole network
        self._n_vectors = int(np.ceil(np.sum(self.structure[1:]) / 128))
        assert self._n_vectors < 3

        self._routing = RoutingGenerator(neuron_size=self._neuron_size, signed_synapses=self._signed_synapses)

    def configure(self, reduce_power=False, initialize=True):
        if initialize:
            init = stadls.ExperimentInit()

            shiftreg = init.shift_register
            shiftreg.select_analog_readout_mux_1_input = shiftreg.AnalogReadoutMux1Input.readout_chain_1
            shiftreg.select_analog_readout_mux_2_input = shiftreg.AnalogReadoutMux2Input.mux_dac_25

            if reduce_power:
                # disable MADC clock
                madc_clock = haldls.PLLClockOutputBlock.ClockOutput()
                madc_clock.enable_output = False
                init.pll_clock_output_block.set_clock_output(halco.PLLClockOutputOnDLS.madc_clk, madc_clock)

                # reduce PPU clock
                ppu_clock = init.pll_clock_output_block.get_clock_output(halco.PLLClockOutputOnDLS.ppu_clk)
                init.adplls[ppu_clock.select_adpll].core_div_m1 = haldls.ADPLL.CoreDivM1(5)

                # reduce supply voltages
                init.dac_channel_block.value[halco.DACChannelOnBoard.vdd25_digital] = 0
                init.dac_channel_block.value[halco.DACChannelOnBoard.vdd25_analog] = 0
                init.dac_channel_block.value[halco.DACChannelOnBoard.vdd12_digital] = 700
                init.dac_channel_block.value[halco.DACChannelOnBoard.vdd12_pll] = 0

                # disable links (two links remaining)
                enable_links = np.zeros(8, dtype=np.bool)
                enable_links[3, 4] = True
                for i, value in enumerate(enable_links):
                    fpga_coord = halco.PhyConfigFPGAOnDLS(i)
                    chip_coord = halco.PhyConfigChipOnDLS(7 - i)
                    init.highspeed_link.common_phy_config_fpga.set_enable_phy(fpga_coord, value)
                    init.highspeed_link.phy_configs_fpga[i].enable_manual_training_mode = not value
                    init.highspeed_link.common_phy_config_chip.set_enable_phy(chip_coord, value)
                    init.highspeed_link.phy_configs_chip[7-i].enable_manual_training_mode = not value
                    init.highspeed_link.phy_configs_chip[7-i].vbias = haldls.PhyConfigChip.VBias(7)

            # TEMP DISABLED FOR TESTING
            # Configuration of DAC for synaptic trace measurement?
            dac_config = init.dac_channel_block
            dac_config.value[halco.DACChannelOnBoard.v_res_meas] = 4095
            dac_config.value[halco.DACChannelOnBoard.mux_dac_25] = 2900

            init_builder, _ = init.generate()
        else:
            init_builder = stadls.PlaybackProgramBuilder()

        # apply external DAC to synapse debug line
        pad_mux = haldls.PadMultiplexerConfig()
        pad_mux.synin_debug_excitatory_to_synapse_intermediate_mux = True
        pad_mux.synin_debug_inhibitory_to_synapse_intermediate_mux = True
        pad_mux.synapse_intermediate_mux_to_pad = True
        init_builder.write(halco.PadMultiplexerConfigOnDLS(1), pad_mux)

        # TEMP ENABLED FOR TESTING
        # Needed for resetting when in fast inference mode (incompatible with simultaneous readout of synaptic traces)
        # channel = haldls.DACChannel()
        # channel.value = int(1.2 / 2.5 * 4095)
        # init_builder.write(halco.DACChannelOnBoard.mux_dac_25, channel)

        # modify neuron config
        boundaries = np.hstack([np.zeros(1, dtype=int), np.array(self.structure[1:]).cumsum()])
        spiking = [True if not isinstance(l, LayerSize) else l.spiking for l in self.structure[1:]] + [False]
        for c in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = self._neuron_calib.neurons[c]

            source = int(c.toEnum()) // self._neuron_size
            population = (source >= boundaries).sum() - 1
            compartment = int(c.toEnum()) % self._neuron_size
            # print(f"{source=} {population=} {compartment=} {spiking[population]=}")
            if compartment == 0:
                if spiking[population]:
                    config.threshold.enable = True
                else:
                    config.threshold.enable = False
                    # Was used during diagnosis of why outputs were spiking
                    # config.event_routing.analog_output = lola.AtomicNeuron.EventRouting.AnalogOutputMode.off
                config.multicompartment.connect_right = True

                config.readout.enable_amplifier = True
                config.readout.enable_buffered_access = False
            else:
                config.membrane_capacitance.capacitance = 0
                config.leak.enable_division = True
                config.readout.enable_amplifier = False
                config.threshold.enable = False
                # Was used during diagnosis of why outputs were spiking
                # config.event_routing.analog_output = lola.AtomicNeuron.EventRouting.AnalogOutputMode.off
            if compartment != (self._neuron_size - 1):
                config.multicompartment.connect_right = True

            # config.threshold.enable = False
            # config.event_routing.analog_output = lola.AtomicNeuron.EventRouting.AnalogOutputMode.off

        # for c in halco.iter_all(halco.AtomicNeuronOnDLS):
        #     config = self._neuron_calib.neurons[c]
        #     print(c, config)

        # apply calibration
        self._cadc_calib.apply(init_builder)
        self._neuron_calib.apply(init_builder)
        # calix.common.cadc.apply_calibration(init_builder, self._cadc_calib)
        # calix.spiking.neuron.apply_calibration(init_builder, self._neuron_calib)

        for c in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            # config = haldls.CommonNeuronBackendConfig()
            config = self._neuron_calib.cocos[c]
            #config.clock_scale_fast = self._neuron_calib.refractory_clock
            #config.clock_scale_slow = self._neuron_calib.refractory_clock
            config.clock_scale_adaptation_pulse = 15
            config.clock_scale_post_pulse = 15
            config.enable_clocks = True
            config.enable_event_registers = True
            for block in range(4):
                config.set_sample_positive_edge(block, True)

            init_builder.write(c, config)

        stadls.run(self._connection, init_builder.done())

        # patch neuron backend configfrom calibration result
        for c in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            calib_neuron_backend = self._neuron_calib.neurons[c.toAtomicNeuronOnDLS()]
            config = self._routing.neuron_backend_configs[c]
            config.refractory_time = calib_neuron_backend.refractory_period.refractory_time
            # print(f"REFRACTORY TIME: {config.refractory_time}", flush=True)
            config.select_input_clock = calib_neuron_backend.refractory_period.input_clock
            config.reset_holdoff = calib_neuron_backend.refractory_period.reset_holdoff

        builder = self._routing.generate()

        # set synapse bias
        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.syn_i_bias_dac, block),

                          haldls.CapMemCell(self.synapse_bias))

        # TEMP DISABLED FOR TESTING
        # set synapse capmem cells
        """
        The correlation measurement is based on discharging a "measurement" capacitor (not the one you read out). 
        It is initially reset to this v_res_meas voltage, that happens automatically at a pre- or postsynaptic spike (for a/causal respectively). 
        The i_bias_ramp controls how fast this capacitor is discharged before the other correlated event arrives, the i_bias_store controls how 
        fast it is discharged after the other event arrived. During the latter phase, the remaining voltage on that measurement capacitor is 
        exponentially weighted transferred to the "storage" capacitor (which you read out).
        So syn_i_bias_store affects how fast the voltage decays during storage, and v_res_meas lets it start off with a lower voltage initially.
        """
        synapse_params = {
                    halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 400,  # Controls time constant of correlation sensor 400 ~ 8 us (smaller values increase time constant)
                    halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 200,  # Amplitude (smaller values increase) 
                    halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 600,  # Ignore
                }

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in synapse_params.items():
                builder.write(halco.CapMemCellOnDLS(k, block),
                              haldls.CapMemCell(v))

        correlation_switch_quad = haldls.ColumnCorrelationQuad()
        switch = correlation_switch_quad.ColumnCorrelationSwitch()
        switch.enable_internal_causal = True
        switch.enable_internal_acausal = True
        """
        enable_cadc_neuron_readout_causal = True
        was never strictly needed, and now should not be used because we need the causal readout for the actual
        synaptic trace
        """
        switch.enable_cadc_neuron_readout_causal = False
        switch.enable_cadc_neuron_readout_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            correlation_switch_quad.set_switch(switch_coord, switch)

        for sq in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            builder.write(sq, correlation_switch_quad)

        for c in halco.iter_all(halco.CommonCorrelationConfigOnDLS):
            ccc = haldls.CommonCorrelationConfig()
            ccc.reset_duration = 5
            builder.write(c, ccc)

        stadls.run(self._connection, builder.done())

        # configure MADC
        builder = stadls.PlaybackProgramBuilder()
        builder.write(halco.CapMemCellOnDLS.readout_ac_mux_i_bias, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_madc_in_500na, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_i_bias, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref, haldls.CapMemCell(400))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_v_ref, haldls.CapMemCell(400))

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp, block),
                          haldls.CapMemCell(70))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_leak_source_follower, block),
                          haldls.CapMemCell(100))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_spike_comparator, block),
                          haldls.CapMemCell(100))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n, block),
                          haldls.CapMemCell(270))

        # static MADC config
        config = haldls.MADCConfig()
        builder.write(halco.MADCConfigOnDLS(), config)

        for coord in halco.iter_all(halco.PadMultiplexerConfigOnDLS):
            builder.write(coord, haldls.PadMultiplexerConfig())

        stadls.run(self._connection, builder.done())

    def set_readout(self, neuron_index: int, target="membrane"):
        neuron_coord = halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(neuron_index * self._neuron_size))

        builder = stadls.PlaybackProgramBuilder()
        mux_config = haldls.ReadoutSourceSelection.SourceMultiplexer()
        if neuron_coord.toNeuronColumnOnDLS() % 2:
            mux_config.neuron_odd[
                neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True
        else:
            mux_config.neuron_even[
                neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True

        config = haldls.ReadoutSourceSelection()
        config.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0),
            mux_config)
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        for c in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = self._neuron_calib.neurons[c]
            config.readout.enable_buffered_access = (c == neuron_coord)
            config.readout.source = getattr(lola.AtomicNeuron.Readout.Source, target)

            builder.write(c.toNeuronConfigOnDLS(), config.asNeuronConfig())

        stadls.run(self._connection, builder.done())

    def write_weights(self, *weights):
        # we from now on assume that we have up to 256 inputs per neuron
        assert self._neuron_size == 2

        shapes = []
        for l, layer in enumerate(self.structure[:-1]):
            next_layer = self.structure[l + 1]
            if not isinstance(next_layer, LayerSize) or not next_layer.recurrent:
                shape = (layer, next_layer)
            else:
                shape = (layer + next_layer, next_layer)
            shapes.append(shape)

        weights_unrolled = np.zeros((256, 256), dtype=int)
        offsets_unrolled = np.zeros((256, 256), dtype=int)

        boundaries = np.hstack([np.zeros(1, dtype=int), np.array(self.structure[1:]).cumsum()])

        assert len(weights) == len(shapes)
        for l, (shape, w, layer) in enumerate(zip(shapes, weights, self.structure[1:])):
            if shape != w.shape:
                msg = f"Shape of weights for layer {layer} is not compatible with the layer specification."
                raise IndexError(msg)

            recurrent = isinstance(layer, LayerSize) and layer.recurrent
            if recurrent:
                a = boundaries[l]
                b = boundaries[l + 1]
                offset = 0

                c = boundaries[l]
                d = boundaries[l + 1]
                weights_unrolled[a:b, c:d] = w[a - b:, :]
                offsets_unrolled[a:b, c:d] = offset

            # non-reccurent weights
            if l == 0:
                a = self._input_shift + 0
                b = self._input_shift + self.structure[0]
                offset = 1
            else:
                a = boundaries[l - 1]
                b = boundaries[l]
                offset = 0

            c = boundaries[l]
            d = boundaries[l + 1]
            weights_unrolled[a:b, c:d] = w[0:b - a, :]
            offsets_unrolled[a:b, c:d] = offset

        # synapses
        weights = np.empty((2, 128, 256), dtype=int)
        weights[0, :, :] = weights_unrolled[:128, :]
        weights[1, :, :] = weights_unrolled[128:, :]
        offsets = np.zeros((2, 128, 256), dtype=int) + 0
        offsets[0, :, :] = offsets_unrolled[:128, :]
        offsets[1, :, :] = offsets_unrolled[128:, :] + (1 << 5)

        synram_top, synram_bottom = self._routing.transform_weights(weights, offsets)

        builder = stadls.PlaybackProgramBuilder()
        builder.write(halco.SynramOnDLS.top, synram_top)
        builder.write(halco.SynramOnDLS.bottom, synram_bottom)
        stadls.run(self._connection, builder.done())

    def extract_measurements(self, *weights, measurements):
        # we from now on assume that we have up to 256 inputs per neuron
        assert self._neuron_size == 2

        shapes = []
        for l, layer in enumerate(self.structure[:-1]):
            next_layer = self.structure[l + 1]
            if not isinstance(next_layer, LayerSize) or not next_layer.recurrent:
                shape = (layer, next_layer)
            else:
                shape = (layer + next_layer, next_layer)
            shapes.append(shape)

        """
        shapes = [
            (25, 243),
            (243, 3),
        ]
        """

        weights_unrolled = np.zeros((256, 256), dtype=int)
        offsets_unrolled = np.zeros((256, 256), dtype=int)

        boundaries = np.hstack([np.zeros(1, dtype=int), np.array(self.structure[1:]).cumsum()])
        """
        boundaries = [0, 243, 246]
        """

        split_measurements = []

        assert len(weights) == len(shapes)
        for l, (shape, w, layer) in enumerate(zip(shapes, weights, self.structure[1:])):
            """
            shape, w, layer = (25, 243), array(25, 243), 243
            shape, w, layer = (243, 3), array(243, 3), 3
            """
            if shape != w.shape:
                msg = f"Shape of weights for layer {layer} is not compatible with the layer specification."
                raise IndexError(msg)

            recurrent = isinstance(layer, LayerSize) and layer.recurrent
            if recurrent:
                a = boundaries[l]
                b = boundaries[l + 1]
                offset = 0

                c = boundaries[l]
                d = boundaries[l + 1]
                weights_unrolled[a:b, c:d] = w[a - b:, :]
                offsets_unrolled[a:b, c:d] = offset

            # non-reccurent weights
            if l == 0:
                a = self._input_shift + 0
                b = self._input_shift + self.structure[0]
                offset = 1
            else:
                a = boundaries[l - 1]
                b = boundaries[l]
                offset = 0

            c = boundaries[l]
            d = boundaries[l + 1]
            """
            self._input_shift = 0

            l = 0
            a, b = 0, 25
            c, d = 0, 243
            offset = 1
            
            a, b = 0, 243
            c, d = 243, 246
            offset = 0
            """
            weights_unrolled[a:b, c:d] = w[0:b - a, :]
            offsets_unrolled[a:b, c:d] = offset

            m = measurements[a:b, c:d].copy()
            split_measurements.append(m)

        """
        Next, the weights_unrolled and offsets_unrolled are dumped into (2, 128, 256) arrays, with offsets having (1 << 5) added to them.
        A call to transform_weights results in synram_top, synram_bottom being returned.

        It starts by computing a shape = (2, 128, 256) and verifies that the weights and sources arrays have that shape.

        Then it swaps the first two axes, resulting in (128, 2, 256), then reshapes it into (128, 512) with Fortran ordering
        """

        
        return split_measurements


    def transform_measurements(self, weights, sources, measurements):

        return
        

    def _signal_ppus(self, builder, coordinate, data):
        for ppu in range(self._n_vectors):
            builder.write(halco.PPUMemoryWordOnDLS(coordinate, halco.PPUOnDLS(ppu)), haldls.PPUMemoryWord(data))
        return builder

    def _measure_correlation_baseline(self):
        # measure correlation baseline
        builder = stadls.PlaybackProgramBuilder()
        gonzales.reset_correlation(builder)
        tickets = gonzales.measure_correlation(builder)
        builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)
        stadls.run(self._connection, builder.done())

        baseline = np.zeros(
                (halco.SynapseRowOnDLS.size, halco.NeuronColumnOnDLS.size),
                dtype=np.int)
        for ticket_id, ticket in enumerate(tickets):
            baseline[ticket_id, :] = ticket.get().causal.to_numpy()
        
        # print(f"Measured causal trace baseline: {baseline.min()} - {baseline.max()}, mean: {np.mean(baseline)}")
        return baseline


    def run(self, input_spikes, n_samples=None, duration=None, measure_power=False, trigger_reset=False, record_madc=False):
        if self._measure_correlation:
            self.baseline = self._measure_correlation_baseline()

        builder = stadls.PlaybackProgramBuilder()

        # configure the number of samples to be recorded
        self._signal_ppus(builder, self._ppu_n_samples_coordinate[0], haldls.PPUMemoryWord.Value(n_samples))

        # let the PPUs reset the batch sample counter
        command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.RESET_BATCH.value))
        self._signal_ppus(builder, self._ppu_signal_coordinate[0], command)

        # enable recurrent connections
        builder.copy_back(enable_recurrency_builder)

        # arm MADC
        if record_madc:
            madc_control = haldls.MADCControl()
            madc_control.enable_power_down_after_sampling = False
            madc_control.start_recording = False
            madc_control.wake_up = True
            madc_control.enable_pre_amplifier = True
            madc_control.enable_continuous_sampling = True

            builder.write(halco.MADCControlOnDLS(), madc_control)

        # sync time
        builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.block_until(halco.TimerOnDLS(), 100)

        if record_madc:
            # trigger MADC sampling
            madc_control.start_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        corr_tickets = []

        timing_offset = 100e-6

        hw_batch_size = len(input_spikes)
        for b in range(hw_batch_size):
            builder.block_until(
                halco.TimerOnDLS(),
                int((b * self.sample_separation + timing_offset) * 1e6 * fisch.fpga_clock_cycles_per_us))
            # print(f"BLOCK UNTIL: {b * self.sample_separation + timing_offset }", flush=True)
            # TEMP DISABLED FOR TESTING
            # start CADC recording via PPU
            if trigger_reset:
                command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.RUN_AND_RESET.value))
            else:
                command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.RUN.value))
            self._signal_ppus(builder, self._ppu_signal_coordinate[0], command)

            if (input_spikes[b][:, 0] >= self.sample_separation).any():
                warnings.warn("Not all spikes are injected within the timing separation window. Expecting faulty timing. Please increase sample separation.")

            times = input_spikes[b][:, 0] + timing_offset + b * self.sample_separation
            labels = input_spikes[b][:, 1].astype(np.int)

            # shift inputs in case the first layer is recurrent
            labels += self._input_shift

            # TEMP DISABLED FOR TESTING
            builder.merge_back(self._routing.generate_spike_train(times, labels))

            # Need to block so that PPU can finish reading out membrane potentials
            builder.block_until(
                halco.TimerOnDLS(),
                int((timing_offset + self.sample_separation * b + 50e-6) * 1e6 * fisch.fpga_clock_cycles_per_us))

            if self._measure_correlation:
                builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)
                tickets = gonzales.measure_correlation(builder)
                # TEMP DISABLED FOR TESTING
                gonzales.reset_correlation(builder)
                corr_tickets.append(tickets)

        builder.block_until(
            halco.TimerOnDLS(),
            int((timing_offset + self.sample_separation * (hw_batch_size + 1)) * 1e6 * fisch.fpga_clock_cycles_per_us))

        if record_madc:
            # stop MADC
            madc_control.start_recording = False
            madc_control.stop_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

        # measure power consumption
        if measure_power:
            tickets = {}
            for ina in halco.iter_all(halco.INA219StatusOnBoard):
                tickets[ina] = builder.read(ina)

        builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)

        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        # disable recurrent connections
        builder.copy_back(disable_recurrency_builder)

        n_vectors = hw_batch_size * n_samples * self._n_vectors
        fpga_mem_ticket = gonzales.get_fpga_memory_ticket(builder, n_vectors)

        duration_tickets = []
        for p in range(2):
            duration_tickets.append(
                    builder.read(halco.PPUMemoryWordOnDLS(self._ppu_duration_coordinate[0], halco.PPUOnDLS(p))))

        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.block_until(halco.TimerOnDLS(), 10000)

        program = builder.done()

        stadls.run(self._connection, program)

        durations = []
        for t in duration_tickets:
            durations.append(int(t.get().value))

        if measure_power:
            total_power = 0.0
            for k, v in tickets.items():
                total_power += v.get().toUncalibratedPower().calculate()
            print(total_power)

        spike_times, spike_labels = self._routing.transform_events_from_chip(program.spikes.to_numpy())

        raw_spikes = np.stack([spike_times, spike_labels]).T
        raw_spikes[:, 0] -= timing_offset

        # group spikes according to layers
        spikes = [[] for l in range(len(self.structure) - 1)]
        for b in range(hw_batch_size):
            b_begin = b * self.sample_separation
            b_end = (b + 1) * self.sample_separation
            mask = (raw_spikes[:, 0] > b_begin) & (raw_spikes[:, 0] < b_end)

            dissected_spikes = raw_spikes[mask, :]

            boundaries = np.hstack([np.zeros(1, dtype=int), np.array(self.structure[1:]).cumsum()])
            for l in range(len(self.structure) - 1):
                layer_mask = (dissected_spikes[:, 1] >= boundaries[l]) & (dissected_spikes[:, 1] < boundaries[l + 1])
                s = dissected_spikes[layer_mask, :]

                # subtract timing offset and population indices
                s[:, 0] -= b_begin
                s[:, 1] -= boundaries[l]
                spikes[l].append(s)

            if (raw_spikes[:, 1] >= boundaries[-1]).any():
                print("Received spikes from unused neurons!")

        # TEMP DISABLED FOR TESTING
        # FPGA
        fpga_data = gonzales.parse_fpga_memory_u8(fpga_mem_ticket)
        trace_data = fpga_data.reshape((hw_batch_size, -1, 128*self._n_vectors))[:, :, ::-1]
        cadc_data = np.stack([trace_data[b, :, :] for b in range(hw_batch_size)]).astype(np.float)

        cadc_data = cadc_data / 256 * 1.2

        traces = []
        for l in range(len(self.structure) - 1):
            traces.append(cadc_data[:, :, boundaries[l]:boundaries[l + 1]])

        causal_traces = []
        if self._measure_correlation:
            inputs = halco.SynapseRowOnSynram.size
            ordering = np.argsort(self._routing._lookup)
            measurement = np.zeros(
                (halco.SynapseRowOnDLS.size, halco.NeuronColumnOnDLS.size),
                dtype=np.int)
            for sample_idx, tickets in enumerate(corr_tickets):
                # b_begin = sample_idx * self.sample_separation + timing_offset
                # first_ticket = int(tickets[0].fpga_time) / fisch.fpga_clock_cycles_per_us / 1e6
                # last_ticket = int(tickets[-1].fpga_time) / fisch.fpga_clock_cycles_per_us / 1e6
                # print(f"Since sample start: {last_ticket - b_begin:.3e}", flush=True)
                # print(f"Correlation readout: {last_ticket - first_ticket:.3e}", flush=True)

                for ticket_id, ticket in enumerate(tickets):
                    measurement[ticket_id, :] = ticket.get().causal.to_numpy()
                # print(f"Raw {sample_idx}: {measurement[0,:]}")
                
                corrected = self.baseline - measurement
                corrected[:inputs, :] = corrected[:inputs, :][ordering, :]
                corrected[inputs:, :] = corrected[inputs:, :][ordering, :]
                # print(f"Shuffled {sample_idx}: {meas[0,:]}")

                causal_traces.append(corrected)

        if record_madc:
            samples = program.madc_samples.to_numpy()
            time = samples["chip_time"][10:] / 125 * 1e-6
            trace = samples["value"][10:].astype(np.float) * 2e-3

            self._madc_samples = np.stack([time, trace]).T

        return spikes, traces, durations, causal_traces

    def load_ppu_program(self, program_path):
        # load PPU program
        elf_file = lola.PPUElfFile(program_path)
        elf_symbols = elf_file.read_symbols()

        self._ppu_n_ppus = elf_symbols["n_ppus"].coordinate
        self._ppu_ppu_id = elf_symbols["ppu_id"].coordinate
        self._ppu_duration_coordinate = elf_symbols["duration"].coordinate
        self._ppu_signal_coordinate = elf_symbols["command"].coordinate
        self._ppu_n_samples_coordinate = elf_symbols["n_samples"].coordinate

        # load and prepare ppu program
        builder = stadls.PlaybackProgramBuilder()

        ppu_control_reg_run = haldls.PPUControlRegister()
        ppu_control_reg_run.inhibit_reset = True

        ppu_control_reg_reset = haldls.PPUControlRegister()
        ppu_control_reg_reset.inhibit_reset = False

        program = elf_file.read_program()
        program_on_ppu = halco.PPUMemoryBlockOnPPU(
            halco.PPUMemoryWordOnPPU(0),
            halco.PPUMemoryWordOnPPU(program.size() - 1)
        )

        for ppu in range(2):
            program_on_dls = halco.PPUMemoryBlockOnDLS(program_on_ppu,
                                                       halco.PPUOnDLS(ppu))

            # ensure PPU is in reset state
            builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_reset)

            # manually initialize memory where symbols will lie, issue #3477
            for _name, symbol in elf_symbols.items():
                value = haldls.PPUMemoryBlock(symbol.coordinate.toPPUMemoryBlockSize())
                symbol_on_dls = halco.PPUMemoryBlockOnDLS(symbol.coordinate,
                                                          halco.PPUOnDLS(ppu))
                builder.write(symbol_on_dls, value)

            builder.write(program_on_dls, program)
            builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_run)

            builder.write(
                halco.PPUMemoryWordOnDLS(self._ppu_n_ppus[0], halco.PPUOnDLS(ppu)),
                haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(self._n_vectors)))

            builder.write(
                halco.PPUMemoryWordOnDLS(self._ppu_ppu_id[0], halco.PPUOnDLS(ppu)),
                haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(ppu)))

        # stop second PPU if it is not used
        if self._n_vectors == 1:
            builder.write(program_on_dls, program)
            builder.write(halco.PPUControlRegisterOnDLS(1), ppu_control_reg_run)
            command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.HALT.value))
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_signal_coordinate[0], halco.PPUOnDLS(1)), command)

        stadls.run(self._connection, builder.done())


if __name__ == "__main__":
    with hxcomm.ManagedConnection() as connection:
        backend = StrobeBackend(connection, calibration="../../../data/calibration/cube_66.npz")
        backend.configure()
        backend.load_ppu_program("../../../../bin/strobe.bin")

        weights = [np.zeros((256, 118), dtype=int), np.zeros((118, 10), dtype=int)]
        weights[0][:, :] = 63
        weights[1][:, :] = 63
        backend.write_weights(*weights)

        isi = 100e-6
        offset = 5e-6
        sources = np.arange(256, 1024)
        times = np.arange(sources.size) * isi + offset

        spikes_hidden, spikes_output, traces = backend.run([np.vstack([times, sources]).T])
        print(spikes_hidden)

        import matplotlib.pyplot as plt
        plt.pcolor(traces[0])
        plt.savefig("traces.pdf")

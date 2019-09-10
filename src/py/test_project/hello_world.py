from dlens_v2 import hal, sta, logger
from dlens_v2.halco import CommonNeuronParameter, NeuronOnDLS, \
    NeuronParameter, SynapseDriverOnDLS, SynapseOnDLS


def main(neuron, num_spikes):
    """
    This script sends a spiketrain to some neuron in bypass mode and reads back
    the spikes received.

    :func:`main` is tested by `test_hello_world`.

    :param neuron: Neuron the spikes are sent to
    :type neuron: NeuronOnDLS
    :param num_spikes: Number of spikes to be sent in
    :type num_spikes: int
    :return: Spikes received
    :rtype: list[hal.RecordedSpike]
    """

    # Logger object
    log = logger.get("HelloWorld")

    # Baseboard and FlySPI configuration
    board = hal.Board()

    # DLSv2 ASIC configuration
    chip = hal.Chip()

    # Synapse driver to be used for sending spikes
    synapse_driver = SynapseDriverOnDLS(0)

    # Static baseboard configuration
    board.set_parameter(board.Parameter.syn_v_bias, 1500)
    board.set_parameter(board.Parameter.capmem_i_buf_bias, 3000)
    board.set_parameter(board.Parameter.capmem_i_ref, 3906)

    # Static configuration of capmem parameters (analog neuron parameters)
    chip.capmem_config.enable_capmem = True
    chip.capmem.set(neuron, NeuronParameter.v_leak, 300)
    chip.capmem.set(neuron, NeuronParameter.v_treshold, 600)
    chip.capmem.set(neuron, NeuronParameter.i_bias_spike_comparator, 650)
    chip.capmem.set(neuron, NeuronParameter.i_spike_comparator_delay, 130)
    chip.capmem.set(neuron, NeuronParameter.i_bias_leak_main, 1022)
    chip.capmem.set(neuron, NeuronParameter.i_bias_leak_sd, 1022)
    chip.capmem.set(neuron, NeuronParameter.i_bias_readout_buffer, 1022)
    chip.capmem.set(neuron, NeuronParameter.i_refractory_time, 300)
    chip.capmem.set(neuron, NeuronParameter.i_bias_exc_syn_input_resistor, 200)
    chip.capmem.set(neuron, NeuronParameter.i_bias_inh_syn_input_resistor, 200)
    chip.capmem.set(CommonNeuronParameter.e_reset, 200)

    # Shared digital neuron configuration
    chip.common_neuron_config.enable_digital_out = True

    # Individual digital neuron configuration
    neuron_config = chip.get_neuron_digital_config(neuron)
    neuron_config.fire_out_mode = neuron_config.FireOutMode.bypass_exc
    neuron_config.mux_readout_mode = neuron_config.MuxReadoutMode.v_mem
    chip.set_neuron_digital_config(neuron, neuron_config)
    chip.enable_buffered_readout(neuron)

    # Synapse RAM configuration
    chip.common_synram_config.pc_conf = 1
    chip.common_synram_config.w_conf = 1
    chip.common_synram_config.wait_ctr_clear = 1

    # Configure Synapses
    synapse_coord = SynapseOnDLS(neuron.toSynapseColumnOnDLS(),
                                 synapse_driver.toSynapseRowOnDLS())
    synapse = chip.get_synapse(synapse_coord)
    synapse.weight = 63
    synapse.address = 42
    chip.set_synapse(synapse_coord, synapse)

    # Set switches between synapses and neurons
    switch = chip.get_column_current_switch(neuron.toColumnCurrentSwitchOnDLS())
    switch.inh_config = switch.Config.disabled
    switch.exc_config = switch.Config.internal
    chip.set_column_current_switch(neuron.toColumnCurrentSwitchOnDLS(), switch)

    # Configure Synapse drivers
    chip.synapse_drivers.pulse_length = 8  # Shared parameter
    chip.synapse_drivers.set_mode(synapse_driver,
                                  chip.synapse_drivers.Mode.excitatory)

    # Create a playback program (all times are in FPGA cycles / 96MHz)
    builder = hal.PlaybackProgramBuilder()
    builder.set_time(0)

    # Create equally spaced spiketrain
    isi = 2000
    for idx in range(num_spikes):
        builder.wait_until(1000 + idx * isi)
        builder.fire(synapse_driver, 42)
    builder.wait_for(1000)
    builder.halt()
    program = builder.done()
    assert isinstance(program, hal.PlaybackProgram)

    ctrl = sta.ExperimentControl()
    ctrl.run_experiment(board, chip, program)

    spikes = program.get_spikes()
    log.INFO("Received spikes: %s" % spikes)

    return spikes


if __name__ == '__main__':
    main(neuron=NeuronOnDLS(0), num_spikes=50)

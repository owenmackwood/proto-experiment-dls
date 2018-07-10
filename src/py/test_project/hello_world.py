import json
import argparse
import pylogging
import pydlsnew as dls
import pydlsnew.coords as coords
import numpy


EXT_SPIKES_ADDRESS = 1


def regular_spiketrain(num_spikes, isi, offset):
    builder = dls.Dls_program_builder()
    builder.set_time(0)
    for spike in range(num_spikes):
        builder.wait_until(offset + spike * isi)
        # First argument is a 32 bit mask which drivers should receive the
        # spike
        builder.fire((1 << 0), EXT_SPIKES_ADDRESS)
    builder.wait_until(2 * offset + num_spikes * isi)
    builder.halt()
    return builder


def make_setup(dac_config, cap_mem_config, pulse_length):
    ret = dls.Setup()

    # Set dac
    ret.board_config.dac_values.from_dict(dac_config)

    # Set cap mem values
    for index in range(coords.Neuron_index.num_neurons):
        ret.chip.cap_mem.neuron_params_from_dict(
            coords.Neuron_index(index),
            cap_mem_config["neuron_params"])
    ret.chip.cap_mem.global_params_from_dict(
        cap_mem_config["global_params"])

    # Set synapse drivers
    ret.chip.syndrv_config.pulse_length(pulse_length)

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dac_config", type=argparse.FileType("r"))
    parser.add_argument("cap_mem_config", type=argparse.FileType("r"))
    # Network parameters
    parser.add_argument("--pulse_length", type=int, default=7)
    parser.add_argument("--weight", type=int, default=63)
    # Timing
    parser.add_argument("--num_spikes", type=int, default=1)
    parser.add_argument("--isi", type=int, default=int(1000))
    parser.add_argument("--offset", type=int, default=int(100000))
    # Debug
    parser.add_argument("--record_out", type=int, default=None)

    args = parser.parse_args()

    # Load dac and cap mem config
    dac_config = json.load(args.dac_config)
    cap_mem_config = json.load(args.cap_mem_config)

    # Setup logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')

    # Create a setup
    setup = make_setup(dac_config, cap_mem_config, args.pulse_length)

    # Set the first row of synapses to the targeted weight
    weights = numpy.zeros(dls.Synram.get_shape(), dtype=int)
    weights[0, :] = args.weight
    setup.chip.synram.set_weights(weights)
    # Set the decoder addresses to listen to external spikes in the first row
    addresses = numpy.zeros(dls.Synram.get_shape(), dtype=int)
    addresses[0, :] = EXT_SPIKES_ADDRESS
    setup.chip.synram.set_addresses(addresses)

    # Set debug output if desired
    if args.record_out is not None:
        neuron = setup.chip.neurons.get(coords.Neuron_index(args.record_out))
        neuron.enable_out(True)
        neuron.mux_readout_mode(dls.Neuron.Mux_readout_mode.vmem)
        setup.chip.neurons.set(coords.Neuron_index(args.record_out), neuron)

    # Create a program with spikes
    program = regular_spiketrain(args.num_spikes, args.isi, args.offset)

    # Execute the experiment
    with dls.connect(dls.get_allocated_board_ids()[0]) as connection:
        setup.do_experiment(connection, program)

    # Print result spikes
    spikes = program.get_spikes()
    if len(spikes) == 0:
        print("There were no spikes :(")

    for spike in program.get_spikes():
        print("Spike from {} at {}".format(spike.address, spike.time))


if __name__ == "__main__":
    main()

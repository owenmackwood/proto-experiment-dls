#!/usr/bin/env python

import pynn_brainscales.brainscales2 as pynn

from template import get_neuron_population


def main():
    """
    Run a minimal pyNN-based experiment: Connect two neuron populations
    and run an emulation. More elaborate examples of using BrainScaleS-2
    through the pyNN-API can be found here:
    https://github.com/electronicvisions/pynn-brainscales/tree/master/brainscales2/examples
    """
    pynn.setup()
    neurons_1 = get_neuron_population(2)
    neurons_2 = get_neuron_population(3)
    pynn.Projection(neurons_1, neurons_2, pynn.AllToAllConnector())
    pynn.run(0.2)
    pynn.end()


if __name__ == "__main__":
    main()

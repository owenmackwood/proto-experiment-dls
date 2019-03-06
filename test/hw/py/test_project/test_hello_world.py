import unittest

from dlens.v2.halco import NeuronOnDLS

from test_project import hello_world


class TestHelloWorld(unittest.TestCase):

    def test_main(self):
        spikes = hello_world.main(neuron=NeuronOnDLS(0), num_spikes=50)

        self.assertEqual(50, len(spikes))
        self.assertTrue(all([spk.neuron == NeuronOnDLS(0) for spk in spikes]))

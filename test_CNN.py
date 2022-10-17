
import unittest
from CNN import CNN, Filter

class t_filter (unittest.TestCase):

    def test_get_size (self):
        self.assertEquals(Filter().size, 1)
        self.assertEquals(Filter(
            size=5
        ).size, 5)

    def test_get_output_size (self):
        self.assertEquals(Filter().output_size((1,1,1)), (1,1,1))
        self.assertEquals(Filter().output_size((5,5,3)), (5,5,3))
        self.assertEquals(Filter(
            count=3
        ).output_size((1,1,1)), (1,1,3))
        self.assertEquals(Filter(
            depth=3
        ).output_size((1,1,3)), (1,1,1))
        self.assertEquals(Filter(
            size=5
        ).output_size((6, 7, 2)), (2, 3, 2))
        self.assertEquals(Filter(
            size=3,
            stride=3
        ).output_size((6, 6, 2)), (2, 2, 2))
        self.assertEquals(Filter(
            padding=1,
            size=3
        ).output_size((2, 2, 2)), (2, 2, 2))

    def test_get_param_counts (self):
        self.assertEquals(Filter().parameter_count_per_node(), 1)
        self.assertEquals(Filter(
            size=5
        ).parameter_count_per_node(), 25)
        self.assertEquals(Filter(
            size=3,
            depth=3
        ).parameter_count_per_node(), 27)
        self.assertEquals(Filter(
            size=3,
            depth=2,
            count=10
        ).parameter_count_per_node(), 18)
        self.assertEquals(Filter(
            padding=2,
            stride=2,
            count=10,
            size=3,
            depth=3
        ).parameter_count_per_node(), 27)

    def test_get_total_param_count (self):
        self.assertEquals(Filter().get_total_parameter_count((1,1,1)), 1)
        self.assertEquals(Filter(
            size=5
        ).get_total_parameter_count((6,6,1)), 25*4)
        self.assertEquals(Filter(
            size=3
        ).get_total_parameter_count((3,3,3)), 9*3)
        self.assertEquals(Filter(
            size=3,
            depth=3
        ).get_total_parameter_count((3,3,3)), 27*1)
        self.assertEquals(Filter(
            size=3,
            depth=3,
            count=10
        ).get_total_parameter_count((3,3,3)), 270)
        self.assertEquals(Filter(
            size=3,
            depth=3,
            count=10,
            padding=1
        ).get_total_parameter_count((3,3,3)), 27*(3*3*10))
        self.assertEquals(Filter(
            size=3,
            depth=3,
            count=10,
            padding=1,
            stride=2
        ).get_total_parameter_count((3,3,3)), 27*(2*2*10))

        


class t_cnn (unittest.TestCase):

    def setUp (self):
        self.c = CNN()

    def test_nothing (self):
        return True

    def test_set_input_size (self):
        ex = (48, 32, 3)
        self.c.set_input_size(ex)
        self.assertEquals(self.c.layerShapes[0], ex)


import unittest
from ImageReader import get_test_images, get_test_labels, get_training_images, get_training_labels
from cv2 import namedWindow, imshow, waitKey, destroyWindow, resizeWindow
from random import sample
from numpy import ndarray

class t_train_images (unittest.TestCase):

    def setUp (self):
        self.imgs = get_training_images()
    
    def test_nothing (self): return True

    def test_length (self): self.assertEquals(len(self.imgs), 60000)

    def test_size (self):
        samp = sample(self.imgs, 4)
        for example in samp:
            self.assertEqual(len(example), 28) # 28 rows
            self.assertEqual(len(example[0]), 28) # 28 columns
            self.assertRaises(TypeError, lambda: len(example[0][0])) # only 1 channel

class t_test_images (t_train_images):

    def setUp (self): self.imgs = get_test_images()

    def test_length(self): self.assertEquals(len(self.imgs), 10000)

class t_train_labels (unittest.TestCase):

    def setUp (self):
        self.labels = get_training_labels()
    
    def test_nothing (self): return True

    def test_length (self):
        self.assertEquals(len(self.labels), 60000)
    
    def test_value (self):
        samp = sample(self.labels, 64)
        for ex in samp:
            self.assertLessEqual(ex, 9)

class t_test_labels (t_train_labels):

    def setUp (self):
        self.labels = get_test_labels()
    
    def test_length(self):
        self.assertEquals(len(self.labels), 10000)

if __name__=="__main__": unittest.main()

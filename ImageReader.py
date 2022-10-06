"""Contains functions for reading in the MNIST data set.
They are expected to be in the working directory."""

from cv2 import Mat
from numpy import ndarray, ubyte    

def get_test_labels() -> list[int]:
    return __get_labels('t10k-labels.idx1-ubyte')

def get_training_labels() -> list[int]:
    return __get_labels('train-labels.idx1-ubyte')

def __get_labels (path) -> list[int]:
    with open(path, 'rb') as fin:
        mnum :int = int.from_bytes(fin.read(4), 'big')
        labelCnt :int = int.from_bytes(fin.read(4), 'big')
        labels = [
            int.from_bytes(fin.read(1), 'big')
            for _ in range(labelCnt)
        ]
    return labels

def get_test_images() -> list[Mat]:
    return __get_images("t10k-images.idx3-ubyte")

def get_training_images() -> list[Mat]:
    return __get_images("train-images.idx3-ubyte")

def __get_images(path)->list[Mat]:
    with open(path, 'rb') as fin:
        mnum :int = int.from_bytes(fin.read(4), 'big')
        imageCnt :int = int.from_bytes(fin.read(4), 'big')
        rows :int = int.from_bytes(fin.read(4), 'big')
        cols :int = int.from_bytes(fin.read(4), 'big')
        images = [
            ndarray(
                shape=(rows, cols),
                dtype=ubyte,
                buffer=fin.read(1 * rows * cols)
            )
            for _ in range(imageCnt)
        ]
    return images

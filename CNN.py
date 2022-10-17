
from math import ceil
from typing import Iterable, Callable
from numpy import ndarray, sqrt, prod, reciprocal
from numpy.random import rand

class Filter:

    def __init__(self, /, size:int=1, count:int=1, stride:int=1, depth:int=1, padding:int=0):
        self.radius :int = (size - 1) // 2
        self.count :int = count
        self.stride :int = stride
        self.depth :int = depth
        self.padding :int = padding
    
    @property
    def size (self)->int: return self.radius*2 + 1

    def parameter_count_per_node (self)->int:
        """Returns the number of parameters, or weights, this filter has per output node"""
        return self.size**2 * self.depth
    
    def get_total_parameter_count (self, input_size:tuple[int,int,int])->int:
        """Returns the total number of parameters this filter would incur with the given input, (minus the bias)"""
        return prod(self.output_size(input_size)) * self.parameter_count_per_node()
    
    def output_size (self, input_size:tuple[int,int,int])->tuple[int,int,int]:
        """Calculates and returns the output shape if this filter was applied to the input shape"""
        assert ((input_size[0]+self.padding*2)-self.size)%self.stride == 0, "This filter does not fit for the given input size"
        assert ((input_size[1]+self.padding*2)-self.size)%self.stride == 0, "This filter does not fit for the given input size"
        assert input_size[2] % self.depth == 0, "Invalid input depth, must at least be a multiple of the filter depth"
        return (
            ceil((input_size[0]+(self.padding-self.radius)*2)/self.stride),
            ceil((input_size[1]+(self.padding-self.radius)*2)/self.stride),
            self.count * input_size[2] // self.depth
        )

    def forward (self, input:ndarray, weights:ndarray)->ndarray:
        shapeOut = self.output_size(input.shape)
        assert weights.shape == shapeOut, self.parameter_count_per_node()
        out = ndarray(shapeOut, input.dtype)
        

    def backward (self):
        pass

class CNN: pass # Pre-declaration
class CNN:
    """Allows the training and testing of a Convolutional Neural Net.
    It is default constructed and then decorated.
    Just about every function is a decorator so explore.
    """

    VALID_INPUT_SIZE =  0x1
    VALID_OUTPUT_SIZE = 0x2
    VALID_ACTIVATOR =   0x4
    VALID_WEIGHT_INITER=0x8
    VALID_ALL =         0xF

    def vf_enable (self, flag): self.valid |= flag
    def vf_disable (self, flag): self.valid &= ~flag

    def __init__(self) -> None:
        self.valid :int = 0
        self.layerShapes :list[tuple[int,int,int]] = [None] * 2
        self.filters :list[Filter] = []
        self.weights :list[ndarray] = []
        self.weightInitName :str = None
        self.weightInits :function = None
        self.activationName :str = None
        self.activation :function = None

    def set_input_size (self, shape:tuple[int,int,int])->CNN:
        """Sets the input layer dimensions"""
        self.layerShapes[0] = shape
        if all(d>0 for d in self.layerShapes[0]):
            self.vf_enable(CNN.VALID_INPUT_SIZE)
        else: raise ValueError(f"The given `shape` ({shape}) had values <= 0 when it shouldn't")
        return self

    def set_activator (self, /, name:str, func:Callable)->CNN:
        """Sets the activation function. Can **either** pass a `name` of a
            common function or pass a custom function `func`.
            The accepted list of functions are listed below.
            The custom function must take a single float and return a float.
            
            Presets
            -------
            - Identity
            - ReLU
            - Leaky ReLU"""
        if name is not None and isinstance(name, str):
            map = {
                "identity": lambda x: x,
                "relu": lambda x: max(0, x),
                "leaky relu": lambda x: max(x*0.01, x)
            }
            if name.lower() in map:
                self.activation = map[name.lower()]
                self.activationName = name
                self.vf_enable(CNN.VALID_ACTIVATOR)
            else:
                raise ValueError(f"Did not find the given `name` ({name}) in the accepted list ({map.keys()})")
        elif func is not None and isinstance(func, function):
            try:
                result = func(0.5) # Test if it's a real function
                assert result is not None and isinstance(result, float)
            except:
                raise ValueError(f"The given function ({func}) is not a valid function. Must be f(x)=y.")
            else:
                self.activation = func
                self.activationName = name or func.__name__
                self.vf_enable(CNN.VALID_ACTIVATOR)
        else:
            raise TypeError("Didn't supply any arguments")
        return self

    def set_weight_initer (self, /, name:str, func:Callable)->CNN:
        if name is not None and isinstance(name, str):
            map = {
                "xavier": lambda i,c: (rand()*2-1)*reciprocal(sqrt(prod(c.layerShapes[i])))
            }
            if name.lower() in map:
                self.weightInitName = name
                self.weightInits = map[name.lower()]
                self.vf_enable(CNN.VALID_WEIGHT_INITER)
            else:
                raise ValueError(f"The given `name` ({name}) was not found the list accepted list ({map})")
        elif func is not None and isinstance(func, function):
            try:
                result = func(0, self)
                assert result is not None and isinstance(result, float)
            except:
                raise ValueError(f"The given function ({func}) is not a valid function. Must be f(i,c)=y, where i is the layer index and c is the CNN.")
            else:
                self.weightInits = func
                self.weightInitName = name or func.__name__
                self.vf_enable(CNN.VALID_WEIGHT_INITER)
        else:
            raise TypeError("Didn't supply any arguments")
        return self

    def check_filters (self)->CNN:
        curShape = self.layerShapes[0]
        for f in self.filters:
            curShape = f.output_size(curShape)

    def add_filter (self, /, size:int, count:int=1, stride:int=1, depth:int=1, padding:int=0)->CNN:
        """Appends a filter and (therefore a layer), to the model."""
        f = Filter(size, count, stride, depth, padding)
        self.filters.append(f)
        self.layerShapes.append(f.output_size(self.layerShapes[-1]))
        self.weights.append(None)
        self.check_filters()
        return self




    def apply_conv (self, input:ndarray, index:int):
        filter :Filter = self.filters[index]
        weights :ndarray = self.weights[index]
        output_shape = filter.output_size(input.shape)    
        
        buffer = bytearray(
            1
            for z in range(output_shape[2])
            for y in range(output_shape[1])
            for x in range(output_shape[0])
        )
        return ndarray(output_shape, input.dtype, buffer)

    def evaluate (self, input:ndarray)->list[int]:
        if not self.valid: raise Exception("Model not valid yet")
        self.layers :list[ndarray] = [None] * len(self.filters)
        i = iter(self.filters)
        f = next(i)

        buffer0 = input @ self.weights[0]
        self.layers.append(ndarray(shape, 'float32', buffer))
        for i, f in enumerate(i):
            shape = f.resultant_size(self.layers[0].shape)
            buffer = self.layers[i] @ self.weights[i+1]
            self.layers.append(ndarray(shape, 'float32', buffer))

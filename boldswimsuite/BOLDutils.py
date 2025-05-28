from __future__ import annotations
import numpy as np
from typing import Union, Optional

GYROMAGNETIC_RATIO = 42.58e6 #Hz/T

class VoxelSize:
    def __init__(self, size: Union[float, np.ndarray, VoxelSize], ndims: Optional[int]=None, grid_shape: Optional[np.ndarray]=None):
        
        if isinstance(size, np.ndarray):
            if len(size.shape) != 1:
                raise ValueError(f'Parameter \'size\' must be either a {ndims} element 1d array.') 
            if ndims is None:
                self._ndims = size.shape[0]
                self._size = size
            else:
                if not isinstance(ndims, int) or ndims <= 0:
                    raise ValueError('Parameter \'ndims\' must be a positive integer.')
                elif size.shape != (ndims,):
                    raise ValueError(f'Incorrect shape for \'size\', should be ({ndims},), got {size.shape}.')
                else:
                    self._size = size
                    self._ndims  = ndims
        
        elif isinstance(size, float):
            self._size = size*np.ones(ndims)
            self._ndims = ndims

        elif isinstance(size, int):
            self._size = float(size)*np.ones(ndims)
            self._ndims = ndims

        elif isinstance(size, VoxelSize):
            self._size = size.value
            self._ndims = size.ndims
        
        else:
            raise TypeError(f'Parameter \'size\' must be either a float, a {ndims} element 1d array or another VoxelSize object.')
        
        self._grid_shape = None

        if grid_shape is not None:
            if isinstance(grid_shape, int):
                self._grid_shape = (grid_shape + np.zeros(self._ndims)).astype(int)
            elif isinstance(grid_shape, np.ndarray):
                if grid_shape.shape != (self._ndims,):
                    raise ValueError(f'Incorrect shape for \'grid_shape\', should be ({ndims},), got {grid_shape.shape}.')
                self._grid_shape = grid_shape.astype(int)
            else:
                raise TypeError(f'Parameter \'grid_shape\' must be either an int or a {ndims} element 1d array.')

    @property
    def value(self) -> np.ndarray:
        return self._size

    @property
    def grid_shape(self) -> np.ndarray:
        return self._grid_shape

    @property
    def dNds(self) -> np.ndarray:
        if self.grid_shape is None:
            return None
        return self.grid_shape/self._size

    @property
    def dsdN(self) -> np.ndarray:
        if self.grid_shape is None:
            return None
        return self._size/self.grid_shape

    @property
    def max(self) -> np.ndarray:
        return 0.5*self._size
    
    @property
    def min(self) -> np.ndarray:
        return -0.5*self._size

    @property
    def xlim(self) -> np.ndarray:
        return np.array([-0.5*self._size[0], 0.5*self._size[0]])

    @property
    def ylim(self) -> np.ndarray:
        return np.array([-0.5*self._size[1], 0.5*self._size[1]])

    @property
    def zlim(self) -> np.ndarray:
        return np.array([-0.5*self._size[2], 0.5*self._size[2]])

    @property
    def ndims(self) -> np.ndarray:
        return self._ndims
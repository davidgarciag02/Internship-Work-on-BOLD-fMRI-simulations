from __future__ import annotations
import numpy as np
from numba import float64, int64, types, boolean   # import the types
from numba.experimental import jitclass


class Offset3D:

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def dBz(self, positions: np.ndarray) -> np.ndarray:
        pass   

class GradientOffset3D(Offset3D):

    def __init__(self, slope: float, gradient_direction: str, active: bool=True):
        
        self.slope = slope

        if gradient_direction not in ('x', 'y', 'z'):
            raise Exception('Gradient direction must be \'x\', \'y\' or \'z\'!')

        self.direction_index = {'x': 0, 'y': 1, 'z': 2}[gradient_direction]
        self.active = active

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def dBz(self, positions: np.ndarray) -> np.ndarray:
        
        if self.active:
            coords = positions[:, self.direction_index]
            dBz = coords*self.slope
            return dBz
        
        else:
            return 0

class ConstantOffset3D(Offset3D):

    def __init__(self, offset: float, active: bool=True):
        
        self.offset = offset
        self.active = active

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def dBz(self, positions: np.ndarray) -> np.ndarray:
        
        if self.active:
            return self.offset
        
        else:
            return 0
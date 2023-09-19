import numpy as np
from typing import Optional, List, Tuple, Union, Literal
from tqdm import tqdm
import warnings
from .BOLDconstants import *
from . import BOLDspins, BOLDgeometry
from scipy import signal as spsig
import scipy as sp

class Sequence:

    def __init__(
        self,
        sample_shape: Union[int, Tuple[int, ...]],
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]]
    ):
        self._curr_step = 0
        self._curr_pulse = 0

        self.pulse_time_indices = pulse_time_indices
        self.pulse_angles = pulse_angles
        self.pulse_axes = pulse_axes

        self.Mx = np.zeros(sample_shape)
        self.My = np.zeros(sample_shape)
        self.Mz = np.ones(sample_shape)

        self.signal = 0 + 0j
        self.EV_signal = 0 + 0j
        self.IV_signal = 0 + 0j

    def get_signals(self, cplx: bool=False) -> Union[Tuple[float, float, float], Tuple[complex, complex, complex]]:

        if cplx:
            return self.signal, self.EV_signal, self.IV_signal

        return np.abs(self.signal), np.abs(self.EV_signal), np.abs(self.IV_signal)

    def step(
        self,
        phase: np.ndarray,
        is_IV: np.ndarray
    ):
        # apply inital pulse to the spins
        if self._curr_step in self.pulse_time_indices:
            self._apply_pulse()
            self._curr_pulse += 1

        self._apply_dephasing(phase)
        self._curr_step += 1

        # calculating the transverse magnetization using the complex notation
        Mxy = self.Mx + 1j * self.My

        Mxy_EV = Mxy[np.logical_not(is_IV)]
        Mxy_IV = Mxy[is_IV]

        self.signal = np.mean(Mxy)
        self.EV_signal = np.mean(Mxy_EV)
        self.IV_signal = np.mean(Mxy_IV)

    def _apply_pulse(
        self,
    ) -> None:

        ax_theta = self.pulse_axes[self._curr_pulse][0],
        ax_phi = self.pulse_axes[self._curr_pulse][1],
        angle = self.pulse_angles[self._curr_pulse],
 
        # https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions

        x = np.array(self.Mx)
        y = np.array(self.My)
        z = np.array(self.Mz)

        u = np.sin(ax_theta) * np.cos(ax_phi)
        v = np.sin(ax_theta) * np.sin(ax_phi)
        w = np.cos(ax_theta)

        cangle = np.cos(angle)
        sangle = np.sin(angle)
        term = (u * x + v * y + w * z) * (1 - cangle)

        self.Mx = u * term + x * cangle + (-w * y + v * z) * sangle
        self.My = v * term + y * cangle + (w * x - u * z) * sangle
        self.Mz = w * term + z * cangle + (-v * x + u * y) * sangle

    def _apply_dephasing(
        self,
        phase: np.ndarray,
    ) -> None:

        x = np.array(self.Mx)
        y = np.array(self.My)

        cphase = np.cos(phase)
        sphase = np.sin(phase)

        self.Mx = x * cphase - y * sphase
        self.My = x * sphase + y * cphase

class SpinSequence(Sequence):

    def __init__(
        self,
        spins: BOLDspins.Spins,
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]]
    ):
        self.spins = spins

        super().__init__(
            sample_shape=spins.num_spins,
            pulse_time_indices=pulse_time_indices,
            pulse_angles=pulse_angles,
            pulse_axes=pulse_axes                
        )

    def step(
        self,
        dt: float
    ):
        self.spins.step(dt=dt)
        phase, vessel_index, dt = self.spins.get_phase_vessel_indices_dt()
        super().step(
            phase=phase, 
            is_IV=vessel_index != 0
        )

    def walk(
        self,
        dt: float,
        num_steps: int,
        progressbar: bool=True
    ):
        if self._curr_step != 0:
            warnings.warn('The walk is not on the 0th step of the sequence!')

        eviv = np.zeros(num_steps) 
        ev = np.zeros(num_steps)
        iv = np.zeros(num_steps)

        text = 'Walking Through'
        for j in tqdm(range(num_steps), desc=text, disable=not progressbar):
            self.step(dt=dt)
            eviv[j], ev[j], iv[j] = self.get_signals(cplx=False)

        return eviv, ev, iv
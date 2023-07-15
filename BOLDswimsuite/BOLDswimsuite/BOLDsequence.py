import numpy as np
from typing import Optional, List, Tuple, Union
from .BOLDconstants import *
from . import BOLDspins

class Sequence:

    def __init__(
        self,
        num_samples: int,
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]],
        dB0: float=0,
    ):
        self._curr_step = 0
        self._curr_pulse = 0

        self.pulse_time_indices = pulse_time_indices
        self.pulse_angles = pulse_angles
        self.pulse_axes = pulse_axes
        self.dB0 = dB0

        self.Mx = np.zeros(num_samples)
        self.My = np.zeros(num_samples)
        self.Mz = np.ones(num_samples)

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
        is_IV: np.ndarray,
        dt: float
    ):

        # time and space independent offset
        phase = self._offset_all(phase, self.dB0, dt)

        # apply inital pulse to the spins
        if self._curr_step in self.pulse_time_indices:
            self._apply_pulse()

        self._apply_dephasing(phase)

        # calculating the transverse magnetization using the complex notation
        Mxy = self.Mx + 1j * self.My

        Mxy_EV = Mxy[np.logical_not(is_IV)]
        Mxy_IV = Mxy[is_IV]

        self.signal = np.mean(Mxy)
        self.EV_signal = np.mean(Mxy_EV)
        self.IV_signal = np.mean(Mxy_IV)

    def spins_step(
        self,
        spins: BOLDspins.Spins,
    ):
        phase, is_IV, dt = spins.get_phase_is_IV_dt()
        self.step(
            phase=phase, 
            is_IV=is_IV, 
            dt=dt
        )

    def _offset_all(
        self,
        phase: np.ndarray,
        dB0: float,
        dt: float
    ) -> np.ndarray:

        phase += 2 * np.pi * GYROMAGNETIC_RATIO * dB0 * (dt * 0.001)

        return phase

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

        self._curr_pulse += 1

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
        self._curr_step += 1
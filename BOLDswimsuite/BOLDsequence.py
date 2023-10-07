import numpy as np
from typing import List, Tuple, Union
from tqdm import tqdm
import warnings
from .BOLDconstants import *
from . import BOLDspins

class Sequence:
    """Object used to define and apply pulse sequences. Holds information about magnetization of the provided dephasing samples. Assumes an initial magnetization of Mx=My=0 and Mz=1.

    Parameters
    ----------
    sample_shape : Union[int, Tuple[int, ...]]
        Shape of the `phase` and `is_IV` arrays that will be given at each time step. If `int`, assumes a 1D array with length `sample_shape`. If `Tuple[int, ...]`, takes this as the array shape.
    pulse_time_indices : List[int]
        List of the number of times steps before each pulse is applied. For example, if it is `[0, 5, 10]`, and each step is 0.2ms, the pulses will be applied at the 0st, 10th and 20th time step, or at t=0ms, 2ms and 4ms. The pulses are always applied before dephasing is applied.
    pulse_angles : List[float]
        List of angle of each pulse (radians). Must be the same length as `pulse_time_indices`
    pulse_axes : List[List[float]]
        List of rotation axes, in polar coordinates (radians). Each axis in the list is a 2-element list with the form `[phi,theta]`. For example, a pulse on the x-axis will be represented as `[np.pi/2, 0]` and a pules on the y-axis will be represented as `[np.pi/2, np.pi/2]`.
    """

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
        """Get the total, extravascular and intravascular signals at the current time step.

        Parameters
        ----------
        cplx : bool, optional
            Whether the signal output should be complex. The default is False, returning the magnitude of the complex signal.

        Returns
        -------
        Union[Tuple[float, float, float], Tuple[complex, complex, complex]]
            3 element Tuple. The first element is the total signal. The second element is the extravascular signal. The third element is the intravascular signal.
        """        

        if cplx:
            return self.signal, self.EV_signal, self.IV_signal

        return np.abs(self.signal), np.abs(self.EV_signal), np.abs(self.IV_signal)

    def step(
        self,
        phase: np.ndarray,
        is_IV: np.ndarray
    ):
        """Advance the sequence by 1 step.

        Parameters
        ----------
        phase : np.ndarray
            Array of dephasing samples.
        is_IV : np.ndarray
            Array indicating whether each phase sample is intravascular or extravascular.
        """

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
    ):
        """Apply the next pulse in the pulse sequence to the magnetization arrays (Mx, My, Mz).
        """        

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
    ):
        """Apply dephasing to the magnetization arrays (Mx, My, Mz), given dephasing samples.

        Parameters
        ----------
        phase : np.ndarray
            Array of dephasing samples.
        """        

        x = np.array(self.Mx)
        y = np.array(self.My)

        cphase = np.cos(phase)
        sphase = np.sin(phase)

        self.Mx = x * cphase - y * sphase
        self.My = x * sphase + y * cphase

class SpinSequence(Sequence):
    """Object used to define and apply pulse sequences to be used with `Spins` objects. Holds information about magnetization of the provided dephasing samples. Assumes an initial magnetization of Mx=My=0 and Mz=1.

    Parameters
    ----------
    spins : BOLDspins.Spins
        Spins object on which the dephasing samples will be taken.
    pulse_time_indices : List[int]
        List of the number of times steps before each pulse is applied. For example, if it is `[0, 5, 10]`, and each step is 0.2ms, the pulses will be applied at the 0st, 10th and 20th time step, or at t=0ms, 2ms and 4ms. The pulses are always applied before dephasing is applied.
    pulse_angles : List[float]
        List of angle of each pulse (radians). Must be the same length as `pulse_time_indices`
    pulse_axes : List[List[float]]
        List of rotation axes, in polar coordinates (radians). Each axis in the list is a 2-element list with the form `[phi,theta]`. For example, a pulse on the x-axis will be represented as `[np.pi/2, 0]` and a pules on the y-axis will be represented as `[np.pi/2, np.pi/2]`.
    """

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
        """Advance the sequence and `Spins` by 1 step.

        Parameters
        ----------
        dt : float
            Length of the time step (ms).
        """

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
        cplx: bool = False,
        progressbar: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the sequence and `Spins` by many steps.

        Parameters
        ----------
        dt : float
            Length of the time steps (ms).
        num_steps : int
            Number of time steps.
        cplx : bool, optional
            Whether the signal output arrays should be complex. The default is False, returning the magnitude of the complex signal.
        progressbar : bool, optional
            3 element Tuple. The first element is the total signal array. The second element is the extravascular signal array. The third element is the intravascular signal array.  

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            _description_
        """
   
        if self._curr_step != 0:
            warnings.warn('The walk is not on the 0th step of the sequence!')

        eviv = np.zeros(num_steps) 
        ev = np.zeros(num_steps)
        iv = np.zeros(num_steps)

        text = 'Walking Through'
        for j in tqdm(range(num_steps), desc=text, disable=not progressbar):
            self.step(dt=dt)
            eviv[j], ev[j], iv[j] = self.get_signals(cplx=cplx)

        return eviv, ev, iv
import numpy as np
from typing import Optional, List, Tuple
from .BOLDconstants import *

class Sequence:

    def __init__(
        self,
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]],
        dB0: float=0,
        dB1fact: float=0,
        T2EV: Optional[float]=None,
        T2IV: Optional[float]=None,
        T1EV: Optional[float]=None,
        T1IV: Optional[float]=None
    ):
        self.pulse_time_indices = pulse_time_indices
        self.pulse_angles = pulse_angles
        self.pulse_axes = pulse_axes
        self.dB0 = dB0
        self.dB1fact = dB1fact

        self.T2EV = T2EV
        self.T1EV = T1EV
        self.T2IV = T2IV
        self.T1IV = T1IV

        self.Mx = None
        self.My = None
        self.Mz = None


    def signal(
        self,
        phase: np.ndarray,
        is_IV: np.ndarray,
        num_samples: int,
        num_dt: int,
        dt: float,
        sample_mask: Optional[np.ndarray]=None,
        cplx: bool=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Calculates the BOLD signal from populated Mx and My arrays.

        Returns
        -------
        signal : array of the signal all all time steps

        '''
        
        self.Mx = np.zeros((num_samples, num_dt + 1))
        self.My = np.zeros((num_samples, num_dt + 1))
        self.Mz = np.zeros((num_samples, num_dt + 1))

        # time and space independent offset
        phase = self.offset_all(phase, self.dB0, dt)

        # set initial Mz for all spins to 1
        self.Mz[:, 0] = 1
        # apply inital pulse to the spins
        self.rot_any(
            self.pulse_axes[0][0],
            self.pulse_axes[0][1],
            self.pulse_angles[0],
            0
        )

        # create a variable with all the pulse intervals (including the end of
        # the simulation)
        intervals = list(self.pulse_time_indices)
        intervals.append(num_dt) 

        for j in range(1, len(intervals)):

            # set the start and end of the intervals
            start = intervals[j - 1] + 1
            end = intervals[j] + 1

            # apply dephasing in the interval
            for i in range(start, end):
                self.rot_phase(phase, i)

            # set the Mz for the interval to be equal to the end of the last
            # interval
            self.Mz[:, start:end + 1] = np.expand_dims(self.Mz[:, start - 1], 1)

            # apply T1 and T2 decay if applicable
            time_range = dt * np.arange(end - start)

            for ind, t in enumerate(time_range):
                # time index
                time_index = ind + start

                # T1 decay
                if self.T2IV is not None:
                    E2 = np.exp(-t / self.T2IV)
                    self.Mx[:, time_index][is_IV[:, time_index - 1] > 0] *= E2
                    self.My[:, time_index][is_IV[:, time_index - 1] > 0] *= E2
                if self.T2EV is not None:
                    E2 = np.exp(-t / self.T2EV)
                    self.Mx[:, time_index][is_IV[:, time_index - 1] == 0] *= E2
                    self.My[:, time_index][is_IV[:, time_index - 1] == 0] *= E2

                # T2 decay
                if self.T1IV is not None:
                    E1 = np.exp(-t / self.T1IV)
                    self.Mz[:, time_index][is_IV[:, time_index - 1] > 0] = \
                        self.Mz[:,time_index][is_IV[:, time_index - 1] > 0] * E1 + (1 - E1)
                if self.T1EV is not None:
                    E1 = np.exp(-t / self.T1EV)
                    self.Mz[:, time_index][is_IV[:, time_index - 1] == 0] = \
                        self.Mz[:,time_index][is_IV[:, time_index - 1] == 0] * E1 + (1 - E1)

            # apply the pulse at the desired time (end of interval) unless the
            # end of the interval is the end of the simulation
            if not (intervals[j] == intervals[-1]):
                # apply dB1 factor
                angle = self.pulse_angles[j] * self.dB1fact
                # apply pulse spin rotation
                self.rot_any(
                    self.pulse_axes[j][0],
                    self.pulse_axes[j][1],
                    angle,
                    end - 1
                )

        # removes spins that are not in the sampling region
        if sample_mask is not None:
            self.Mx = self.Mx * sample_mask
            self.My = self.My * sample_mask
        else:
            self.Mx = self.Mx
            self.My = self.My

        # calculating the transverse magnetization using the complex notation
        Mxy = self.Mx + 1j * self.My

        is_IV_mask = np.concatenate(
            (np.expand_dims(is_IV[:,0],1) > 0, is_IV > 0), 1)

        Mxy_EV = Mxy * np.logical_not(is_IV_mask)
        Mxy_IV = Mxy * is_IV_mask

        # averaging the magnetization over the spins at all time steps
        if sample_mask is not None:
            complex_signal = np.sum(Mxy, 0) / np.count_nonzero(sample_mask, 0)
            complex_signal_EV = np.sum(Mxy_EV, 0) / np.count_nonzero(np.logical_not(is_IV_mask) * sample_mask, 0)
            complex_signal_IV = np.sum(Mxy_IV, 0) / np.count_nonzero(is_IV_mask * sample_mask, 0)
        else:
            complex_signal = np.mean(Mxy, 0)
            complex_signal_EV = np.mean(Mxy_EV, 0)
            complex_signal_IV = np.mean(Mxy_IV, 0)

        if cplx:
            return complex_signal, complex_signal_EV, complex_signal_IV
        else:
            # taking the magnitude of the complex signal
            signal = np.abs(complex_signal)
            signal_EV = np.abs(complex_signal_EV)
            signal_IV = np.abs(complex_signal_IV)
            return signal, signal_EV, signal_IV

    def offset_all(
        self,
        phase: np.ndarray,
        dB0: float,
        dt: float
    ) -> None:
        '''Adds an extra bulk magnetic offset to all spin phases

        Parameters
        ----------
        dB0 : float, bulk magnetic field offset (T)

        Returns
        -------
        None.

        '''
        phase += 2 * np.pi * GYROMAGNETIC_RATIO * dB0 * (dt * 0.001)

        return phase

    def rot_any(
        self,
        ax_theta: float,
        ax_phi: float,
        angle: float,
        time_index: int
    ) -> None:
 
        # https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions

        x = np.array(self.Mx[:, time_index])
        y = np.array(self.My[:, time_index])
        z = np.array(self.Mz[:, time_index])

        u = np.sin(ax_theta) * np.cos(ax_phi)
        v = np.sin(ax_theta) * np.sin(ax_phi)
        w = np.cos(ax_theta)

        cangle = np.cos(angle)
        sangle = np.sin(angle)
        term = (u * x + v * y + w * z) * (1 - cangle)

        self.Mx[:, time_index] = u * term + x * cangle + (-w * y + v * z) * sangle
        self.My[:, time_index] = v * term + y * cangle + (w * x - u * z) * sangle
        self.Mz[:, time_index] = w * term + z * cangle + (-v * x + u * y) * sangle

    def rot_phase(
        self,
        phase: np.ndarray,
        time_index: int
    ) -> None:
        '''Rotates spin magnetization at a specific time step around the z axis (precession)

        Parameters
        ----------
        tInd : integer, time step index

        Returns
        -------
        None.

        '''
        x = np.array(self.Mx[:, time_index - 1])
        y = np.array(self.My[:, time_index - 1])
        phases = phase[:, time_index - 1]

        cphase = np.cos(phases)
        sphase = np.sin(phases)

        self.Mx[:, time_index] = x * cphase - y * sphase
        self.My[:, time_index] = x * sphase + y * cphase
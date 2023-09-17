from . import BOLDgeometry, BOLDsequence
from .BOLDconstants import *
from scipy import signal as spsig
import scipy as sp
import numpy as np
from tqdm import tqdm
import warnings
from typing import List, Literal, Optional

class DeterministicDiffuser2D(BOLDsequence.Sequence):
    
    def __init__(
        self,
        geometry: BOLDgeometry.DiscreteVoxel2D,
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]],
        ADC: float,
        dt: float,
        kernel_type: Literal['ModifiedBessel', 'Gaussian']='ModifiedBessel',
        permeable_vessels: bool=False
    ):
        self.ADC = ADC
        self.dt = dt
        self.geometry = geometry
        self.kernel_type = kernel_type
        self.permeable_vessels = permeable_vessels

        self.kernel = self._make_kernel()

        super().__init__(
            sample_shape=geometry.dBz_grid.shape,
            pulse_time_indices=pulse_time_indices,
            pulse_angles=pulse_angles,
            pulse_axes=pulse_axes                
        )

    def step(
        self,
        dt: Optional[float]=None
    ):
        
        dt = self.dt if dt is None else dt

        if dt != self.dt:
            self.dt=dt
            self.kernel = self._make_kernel()

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001

        phase = self.geometry.dBz_grid * phase_conversion_factor

        is_IV = (self.geometry.vessel_index_grid > 0)
        not_is_IV = np.logical_not(is_IV)

        # apply inital pulse to the spins
        if self._curr_step in self.pulse_time_indices:
            self._apply_pulse()
            self._curr_pulse += 1

        self._apply_dephasing(phase)
        self._curr_step += 1

        if self.permeable_vessels:
            self.Mx, self.My = self._diffusion_convolution_permeable(self.Mx, self.My, self.kernel, self.kernel_type) 
        else:
            self.Mx, self.My = self._diffusion_convolution_impermeable(self.Mx, self.My, is_IV, not_is_IV, self.kernel, self.kernel_type) 

        # calculating the transverse magnetization using the complex notation
        Mxy = self.Mx + 1j * self.My

        Mxy_EV = Mxy[np.logical_not(is_IV)]
        Mxy_IV = Mxy[is_IV]

        self.signal = np.mean(Mxy)
        self.EV_signal = np.mean(Mxy_EV)
        self.IV_signal = np.mean(Mxy_IV)
    
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
    
    def _make_kernel(self) -> np.ndarray:
        
        sigma_sq = self.ADC*2*self.dt*0.001

        grid_range = np.linspace(-self.geometry.size/2, self.geometry.size/2, self.geometry.N)

        X, Y = np.meshgrid(grid_range, grid_range)

        if self.kernel_type == 'Gaussian':
            kernel = np.exp(-((X ** 2 +  Y ** 2) / (2 * sigma_sq)))
            kernel /= np.sum(kernel)
        
        elif self.kernel_type == 'ModifiedBessel':
            sigma = np.sqrt(sigma_sq)
            dx = self.geometry.size/self.geometry.N
            Nhw = int(np.ceil(6*sigma/dx))
            kernel_range = np.abs(np.arange(-Nhw, Nhw+1))
            kernel = np.exp(-(sigma/dx)**2)*sp.special.iv(kernel_range, (sigma/dx)**2)   
        
        return kernel
    
    @staticmethod
    def _diffusion_convolution_permeable(
        Mx: np.ndarray, 
        My: np.ndarray,
        kernel: np.ndarray,
        kernel_type: Literal['ModifiedBessel', 'Gaussian']
    ):
        if kernel_type == 'Gaussian':
            Mx = spsig.fftconvolve(Mx, kernel, mode='same')
            My = spsig.fftconvolve(My, kernel, mode='same')
       
        elif kernel_type == 'ModifiedBessel':
            Mx = sp.ndimage.convolve1d(Mx, kernel, axis=0, mode='wrap')
            Mx = sp.ndimage.convolve1d(Mx, kernel, axis=1, mode='wrap')
            
            My = sp.ndimage.convolve1d(My, kernel, axis=0, mode='wrap')
            My = sp.ndimage.convolve1d(My, kernel, axis=1, mode='wrap')
        
        return Mx, My
    
    @staticmethod
    def _diffusion_convolution_impermeable(
        Mx: np.ndarray, 
        My: np.ndarray, 
        ip_map: np.ndarray, 
        ep_map: np.ndarray,
        kernel: np.ndarray,
        kernel_type: Literal['ModifiedBessel', 'Gaussian']
    ):
        
        conep, conip = DeterministicDiffuser2D._diffusion_convolution_permeable(ep_map * 1.0, ip_map * 1.0, kernel, kernel_type)

        Wep = conip * ep_map
        Wip = conep * ip_map
        Mxep = Mx * ep_map 
        Myep = My * ep_map
        Mxip = Mx * ip_map
        Myip = My * ip_map
             
        Mxep_diff, Myep_diff = DeterministicDiffuser2D._diffusion_convolution_permeable(Mxep, Myep, kernel, kernel_type)
        Mxep = Mxep_diff * ep_map + Wep * Mxep
        Myep = Myep_diff * ep_map + Wep * Myep
        
        Mxip_diff, Myip_diff = DeterministicDiffuser2D._diffusion_convolution_permeable(Mxip, Myip, kernel, kernel_type)
        Mxip = Mxip_diff * ip_map + Wip * Mxip
        Myip = Myip_diff * ip_map + Wip * Myip

        Mx = Mxep + Mxip
        My = Myep + Myip
        
        return Mx, My
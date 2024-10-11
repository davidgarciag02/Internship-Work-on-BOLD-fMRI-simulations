from . import BOLDgeometry, BOLDsequence
from .BOLDconstants import *
from scipy import signal as spsig
import scipy as sp
import numpy as np
from tqdm import tqdm
import warnings
from typing import List, Literal, Optional, Tuple

class DeterministicDiffuser2D(BOLDsequence.Sequence):    
    """Object for applying a pulse sequence through deterministic diffusion. 

    Parameters
    ----------
    geometry : BOLDgeometry.DiscreteVoxel2D
        2D discrete voxel to apply the deterministic diffusion to. Note that the vessel permeabilities defined in this object are superseded by the `permeable_vessels` parameter.
    pulse_time_indices : List[int]
        List of the number of times steps before each pulse is applied. For example, if it is `[0, 10, 20]`, and each step is 0.2ms, the pulses will be applied at the 0st, 10th and 20th time step, or at t=0ms, 2ms and 4ms. The pulses are always applied before dephasing is applied.
    pulse_angles : List[float]
        List of angle of each pulse (radians). Must be the same length as `pulse_time_indices`
    pulse_axes : List[List[float]]
        List of rotation axes, in polar coordinates (radians). Each axis in the list is a 2-element list with the form `[phi,theta]`. For example, a pulse on the x-axis will be represented as `[np.pi/2, 0]` and a pules on the y-axis will be represented as `[np.pi/2, np.pi/2]`.
    ADC : float
        Apparent diffusion coefficient (mm^2/s).
    dt : float
        Time step length for the inital phase calculation (ms).
    T2EV: Optional[float]
        T2 value for the extravascular space (ms). By default no T2EV is applied.
    T2IV: Optional[float]
        T2 value for the intravascular space (ms). By default no T2IV is applied.
    T1EV: Optional[float]
        T1 value for the extravascular space (ms). By default no T1EV is applied.
    T1IV: Optional[float]
        T1 value for the intravascular space (ms). By default no T1IV is applied.
    kernel_type : Literal['ModifiedBessel', 'Gaussian']
        The type of convolution kernel to use. Default is 'ModifiedBessel'.
    permeable_vessels : bool
        If False, will use a correction method to stop diffusion across vessel walls. Otherwise vessels will be permeable. Default is False.
    """

    def __init__(
        self,
        geometry: BOLDgeometry.DiscreteVoxel2D,
        pulse_time_indices: List[int],
        pulse_angles: List[float],
        pulse_axes: List[List[float]],
        ADC: float,
        dt: float,
        T2EV: Optional[float]=None,
        T2IV: Optional[float]=None,
        T1EV: Optional[float]=None,
        T1IV: Optional[float]=None,
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
            pulse_axes=pulse_axes,
            T2EV=T2EV,
            T2IV=T2IV,
            T1EV=T1EV,
            T1IV=T1IV           
        )

    def step(
        self,
        dt: Optional[float]=None
    ):
        """Advance the diffuser by 1 step.

        Parameters
        ----------
        dt : Optional[float]
            Length of the time step (ms). If not provided, will use the same time step length as the previous step (or as defined during object instantiation).
        """
        
        dt = self.dt if dt is None else dt

        # if the time step length is changed, the kernel has to be remade
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
            self.Mx, self.My = self._unrestricted_diffusion_convolution(self.Mx, self.My, self.kernel, self.kernel_type) 
        else:
            self.Mx, self.My = self._restricted_diffusion_convolution(self.Mx, self.My, is_IV, not_is_IV, self.kernel, self.kernel_type) 

        if self.T2EV is not None:
            E2 = np.exp(-dt/self.T2EV)
            self.Mx[np.logical_not(is_IV)] *= E2
            self.My[np.logical_not(is_IV)] *= E2
        if self.T2IV is not None:
            E2 = np.exp(-dt/self.T2IV)
            self.Mx[is_IV] *= E2
            self.My[is_IV] *= E2
        if self.T1EV is not None:
            E1 = np.exp(-dt/self.T1EV)
            self.Mz[np.logical_not(is_IV)] = self.Mz[np.logical_not(is_IV)]*E1+(1-E1)
        if self.T1IV is not None:
            E2 = np.exp(-dt/self.T1IV)
            self.Mz[is_IV] = self.Mz[is_IV]*E1+(1-E1)

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
        cplx: bool = False,
        progressbar: bool=True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the diffuser by many steps.

        Parameters
        ----------
        dt : float
            Length of the time steps (ms).
        num_steps : int
            Number of time steps.
        cplx : bool, optional
            Whether the signal output arrays should be complex. The default is False, returning the magnitude of the complex signal.
        progressbar : bool, optional
            Whether to show a progress bar in the terminal, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            3 element Tuple. The first element is the total signal array. The second element is the extravascular signal array. The third element is the intravascular signal array.
        """

        if self._curr_step != 0:
            warnings.warn('The walk is not on the 0th step of the sequence!')

        dtype = float if not cplx else complex

        eviv = np.zeros(num_steps, dtype=dtype) 
        ev = np.zeros(num_steps, dtype=dtype)
        iv = np.zeros(num_steps, dtype=dtype)

        text = 'Walking Through'
        for j in tqdm(range(num_steps), desc=text, disable=not progressbar):
            self.step(dt=dt)
            eviv[j], ev[j], iv[j] = self.get_signals(cplx=cplx)

        return eviv, ev, iv
    
    def _make_kernel(self) -> np.ndarray:
        """Create the diffusion kernel according to `self.kernel_type`.

        Returns
        -------
        np.ndarray
            The generated kernel.
        """        

        sigma_sq = self.ADC*2*self.dt*0.001

        if self.kernel_type == 'Gaussian':
            grid_range = np.linspace(-self.geometry.size/2, self.geometry.size/2, self.geometry.N)
            X, Y = np.meshgrid(grid_range, grid_range)
            kernel = np.exp(-((X ** 2 +  Y ** 2) / (2 * sigma_sq)))
            kernel /= np.sum(kernel)
        
        elif self.kernel_type == 'ModifiedBessel':
            sigma = np.sqrt(sigma_sq)
            dx = self.geometry.size/self.geometry.N
            Nhw = int(np.ceil(6*sigma/dx))
            kernel_range = np.abs(np.arange(-Nhw, Nhw+1))
            kernel = np.exp(-(sigma/dx)**2)*sp.special.iv(kernel_range, (sigma/dx)**2)   
        
            kernel_size = len(kernel)
            if kernel_size < 5:
                warnings.warn(f'Diffusion kernel size is very small (size={kernel_size}). This will likely introduce significant error in the diffusion! Fix by increasing spatial resolution of the voxel (N).')

            else:
                tqdm.write(f'Diffuson kernel size: {kernel_size}')

        return kernel
    
    @staticmethod
    def _unrestricted_diffusion_convolution(
        Mx: np.ndarray, 
        My: np.ndarray,
        kernel: np.ndarray,
        kernel_type: Literal['ModifiedBessel', 'Gaussian']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 1 step of unrestricted deterministic diffusion to the magnetization arrays (Mx, My).

        Parameters
        ----------
        Mx : np.ndarray
            Array of magnetization in the x-direction.
        My : np.ndarray
            Array of magnetization in the y-direction.
        kernel : np.ndarray
            Diffusion kernel.
        kernel_type : Literal['ModifiedBessel', 'Gaussian']
            The type of convolution kernel to use. Default is 'ModifiedBessel'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The magnetization arrays (Mx, My) with 1 step of diffusion applied.
        """    
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
    def _restricted_diffusion_convolution(
        Mx: np.ndarray, 
        My: np.ndarray, 
        ip_map: np.ndarray, 
        ep_map: np.ndarray,
        kernel: np.ndarray,
        kernel_type: Literal['ModifiedBessel', 'Gaussian']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 1 step of restricted deterministic diffusion to the magnetization arrays (Mx, My).

        Parameters
        ----------
        Mx : np.ndarray
            Array of magnetization in the x-direction.
        My : np.ndarray
            Array of magnetization in the y-direction.
        ip_map : np.ndarray
            Boolean array indicating whether each magnetization sample is intravascular.
        ep_map : np.ndarray
            Boolean array indicating whether each magnetization sample is extravascular.
        kernel : np.ndarray
            Diffusion kernel.
        kernel_type : Literal['ModifiedBessel', 'Gaussian']
            The type of convolution kernel to use. Default is 'ModifiedBessel'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The magnetization arrays (Mx, My) with 1 step of diffusion applied.
        """               
        conep, conip = DeterministicDiffuser2D._unrestricted_diffusion_convolution(ep_map * 1.0, ip_map * 1.0, kernel, kernel_type)

        Wep = conip * ep_map
        Wip = conep * ip_map
        Mxep = Mx * ep_map 
        Myep = My * ep_map
        Mxip = Mx * ip_map
        Myip = My * ip_map
             
        Mxep_diff, Myep_diff = DeterministicDiffuser2D._unrestricted_diffusion_convolution(Mxep, Myep, kernel, kernel_type)
        Mxep = Mxep_diff * ep_map + Wep * Mxep
        Myep = Myep_diff * ep_map + Wep * Myep
        
        Mxip_diff, Myip_diff = DeterministicDiffuser2D._unrestricted_diffusion_convolution(Mxip, Myip, kernel, kernel_type)
        Mxip = Mxip_diff * ip_map + Wip * Mxip
        Myip = Myip_diff * ip_map + Wip * Myip

        Mx = Mxep + Mxip
        My = Myep + Myip
        
        return Mx, My

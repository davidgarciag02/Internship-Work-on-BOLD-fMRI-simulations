import numpy as np
from scipy import fft, io
from tqdm import tqdm
import os
from typing import List, Dict, Optional
from .BOLDconstants import *
from . import BOLDvoxel

class Grid3D:

    def __init__(
        self,
        B0: float,
        dt: float,
        N: int,
    ):
        self.mask: np.ndarray = None
        self.phase: np.ndarray = None
        self.permeation_probability_list: List[float] = None
        self.dchi_list: List[float] = None

        self.N = N
        self.B0 = B0
        self.dt = dt

        self.size = None

    def populate(
        self,
        mode: str='FFTfromVoxel',
        size: Optional[float] = None,
        padding: Optional[float]=None,
        id_permeation_probabilities: Optional[Dict[int, float]]=None,
        id_dchis: Optional[Dict[int,float]]=None,
        voxel: BOLDvoxel.Voxel3D=None,
        file: str=None,
        extend: bool=False,
        progressbar: bool=True
    ):
        if mode == 'AnalyticalFromVoxel':
            self._generate_grid_analytical_from_voxel(voxel, progressbar)

        elif mode == 'FFTFromVoxel':
            self.padding = padding
            self._generate_grid_FFT_from_voxel(voxel, extend, progressbar)

        elif mode == 'FFTFromFile':
            self.size = size
            self.padding = padding
            self.permeation_probability_list = [id_permeation_probabilities[i] for i in id_permeation_probabilities] if id_permeation_probabilities is not None else None
            self.dchi_list = [id_dchis[i] for i in id_dchis]
            self._generate_grid_FFT_from_file(file) 

        elif mode == 'MaskFromVoxel':
            self._generate_mask_analytical_from_voxel(voxel)
        else:
            raise Exception(f'\'{mode}\' is not a valid mode!')

    @staticmethod
    def dchi_mask_to_phase_FFT(
        grid_dchi: np.ndarray,
        padding: int,
        dt: float,
        B0: float
    ):
        N = grid_dchi.shape[0]
        half_N = int(np.ceil(N/2))

        if N % 2 != 0:
            pos_range = np.linspace(-half_N + 1, half_N - 1, N) 
        else:
            pos_range = np.linspace(-half_N + 1, half_N, N)
        
        X, Y, Z = np.meshgrid(pos_range, pos_range, pos_range)
        r_squared = X**2 + Y**2 + Z**2

        zeros_indices = np.where(r_squared == 0)
        r_squared[zeros_indices] = 1 #dummy value to prevent divide by 0
        kernel_pos = (1/(4*np.pi)) * (3*Z**2 - r_squared) / (r_squared**(5/2))

        kernel_pos[zeros_indices] = 0 #set divide by 0 value to 0

        pad = padding * 2 + N

        kernel_FFT = fft.rfftn(kernel_pos, s=(pad,pad,pad))

        susceptibility_map_FFT = fft.rfftn(4*np.pi*grid_dchi, s=(pad,pad,pad))

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * dt * B0 * 0.001
        phase_padded = phase_conversion_factor * (
            fft.irfftn(
                susceptibility_map_FFT * kernel_FFT, 
                s=(pad, pad, pad), 
                overwrite_x = True
            )
        )

        shift = int(np.floor(N/2) - 1) - padding
        phase_padded = np.roll(phase_padded, shift=(-shift, -shift, -shift), axis=(0,1,2))
        
        #import matplotlib.pyplot as plt
        #plt.imshow(phase_padded[:,:,100])
        #plt.show()
        if padding > 0:
            phase = phase_padded[padding:-padding,padding:-padding,padding:-padding]
        else:
            phase = phase_padded

        return phase

    def _generate_grid_FFT_from_file(
        self,
        filepath: str
    ):  
        
        filename, file_extension = os.path.splitext(filepath)

        if file_extension == '.txt':
            mask = np.loadtxt(filepath)

            if not mask.shape[0]**2 == mask.shape[1]:
                raise Exception('Input txt mask is not isometric!')

            self.mask = mask.reshape((mask.shape[0], mask.shape[0], mask.shape[0]))

        elif file_extension == '.npy':
            self.mask = np.load(filepath)

            if not all([N == self.mask.shape[0] for N in self.mask.shape]) and len(self.mask.shape) != 3:
                raise Exception('Input .npy mask is not isometric!')

        elif file_extension == '.mat':
            mask_dict = io.loadmat(filepath, mdict=None, appendmat=True)
        
            self.mask = mask_dict['mask']

            if not all([N == self.mask.shape[0] for N in self.mask.shape]) and len(self.mask.shape) != 3:
                raise Exception('Input MATLAB mask is not isometric!')
        
        else:
            raise Exception('File did not have a supported file type (i.e.: .txt, .npy, .mat)')
        
        if self.mask.shape[0] != self.N:
            raise Exception('Input mask \'N\' is not the same as the grid \'N\'!')

        grid_dchi = np.array(self.mask, dtype=float)

        for i, dchi in enumerate(self.dchi_list):
            grid_dchi[grid_dchi == i+1] = dchi
        
        self.phase = self.dchi_mask_to_phase_FFT(grid_dchi, self.N, self.padding, self.dt)

    def _generate_grid_FFT_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel3D,
        extend: bool,
        progressbar: bool
    ):
        self.size = voxel.size

        N = self.N + 2*self.padding if extend else self.N

        self.mask = np.zeros((N,N,N), dtype=int)
        grid_dchi = np.zeros((N,N,N))

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            subvox_size = self.size/self.N
            is_IV = vsl.grid_is_IV(N, subvox_size)

            mask_tmp = np.logical_and(np.logical_not(self.mask), is_IV)
            
            self.mask[mask_tmp] = vsl_counter + 1
            grid_dchi[mask_tmp] = vsl.dchi
        
        self.permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        if extend:
            self.phase = self.dchi_mask_to_phase_FFT(grid_dchi, 0, self.B0, self.dt)
        else:
            self.phase = self.dchi_mask_to_phase_FFT(grid_dchi, self.padding, self.B0, self.dt)

        if extend and self.padding > 0:
            self.mask = self.mask[self.padding:-self.padding,self.padding:-self.padding,self.padding:-self.padding]
            self.phase = self.phase[self.padding:-self.padding,self.padding:-self.padding,self.padding:-self.padding]

    def _generate_grid_analytical_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel3D,
        progressbar: bool
    ):
        self.size = voxel.size

        linear_coord = np.linspace(-self.size/2, self.size/2, self.N)
        X, Y, Z = np.meshgrid(linear_coord, linear_coord, linear_coord)

        linear_pos = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        linear_mask = np.zeros(self.N**3)
        linear_offset = np.zeros(self.N**3)

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            is_IV, dBz_EV, dBz_IV = vsl.dBz_mask_from_positions(linear_pos, self.B0)
            
            is_IV = np.logical_and(is_IV, (linear_mask == 0))
            linear_mask[is_IV] = vsl_counter + 1

            linear_offset += is_IV * dBz_IV + np.logical_not(is_IV) * dBz_EV

        self.mask = linear_mask.reshape(self.N,self.N,self.N)

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001
        self.phase = phase_conversion_factor * linear_offset.reshape(self.N,self.N,self.N)

        self.permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

    def _generate_mask_analytical_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel3D,
        progressbar: bool
    ):
        self.mask = np.zeros((self.N,self.N,self.N))

        text = 'Creating Mask'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            subvox_size = self.size/self.N
            is_IV = vsl.grid_is_IV(self.N, subvox_size)

            mask_tmp = np.logical_and(np.logical_not(self.mask), is_IV)
            
            self.mask[mask_tmp] = vsl_counter + 1

class Grid2D:

    def __init__(
        self,
        B0: float,
        dt: float,
        N: int,
    ):
        self.mask: np.ndarray = None
        self.phase: np.ndarray = None
        self.permeation_probability_list: List[float] = None
        self.dchi_list: List[float] = None

        self.N = N
        self.B0 = B0
        self.dt = dt

        self.size = None

    def populate(
        self,
        mode: str='FFTfromVoxel',
        size: Optional[float] = None,
        padding: Optional[float]=None,
        id_permeation_probabilities: Optional[Dict[int, float]]=None,
        id_dchis: Optional[Dict[int,float]]=None,
        voxel: Optional[BOLDvoxel.Voxel3D]=None,
        file: Optional[str]=None,
        extend: bool=False,
        progressbar: bool=True
    ):
        if mode == 'AnalyticalFromVoxel':
            self._generate_grid_analytical_from_voxel(voxel, progressbar)

        elif mode == 'FFTFromVoxel':
            self.padding = padding
            self._generate_grid_FFT_from_voxel(voxel, extend, progressbar)

        elif mode == 'FFTFromFile':
            self.size = size
            self.padding = padding
            self.permeation_probability_list = [id_permeation_probabilities[i] for i in id_permeation_probabilities] if id_permeation_probabilities is not None else None
            self.dchi_list = [id_dchis[i] for i in id_dchis.keys]
            self._generate_grid_FFT_from_file(file) 

        elif mode == 'MaskFromVoxel':
            self._generate_mask_analytical_from_voxel(voxel)
        else:
            raise Exception(f'\'{mode}\' is not a valid mode!')

    @staticmethod
    def dchi_mask_to_phase_FFT(
        grid_dchi: np.ndarray,
        N: int,
        padding: int,
        dt: float,
        B0: float
    ):
        #FIXME This does not work
        raise Exception('FFT methods not currently working!')
        half_N = int(np.ceil(N/2))

        if N % 2 != 0:
            pos_range = np.linspace(-half_N + 1, half_N - 1, N) 
        else:
            pos_range = np.linspace(-half_N + 1, half_N, N)
        
        X, Y = np.meshgrid(pos_range, pos_range)
        r_squared = X**2 + Y**2

        zeros_indices = np.where(r_squared == 0)
        r_squared[zeros_indices] = 1 #dummy value to prevent divide by 0
        kernel_pos = -(1/(4*np.pi)) * r_squared / (r_squared**(5/2))

        kernel_pos[zeros_indices] = 0 #set divide by 0 value to 0

        pad = padding * 2 + N

        kernel_FFT = fft.rfftn(kernel_pos, s=(pad,pad))

        susceptibility_map_FFT = fft.rfftn(4*np.pi*grid_dchi, s=(pad,pad))

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * dt * B0 * 0.001
        phase_padded = phase_conversion_factor * (
            fft.irfftn(
                susceptibility_map_FFT * kernel_FFT, 
                s=(pad, pad), 
                overwrite_x = True
            )
        )

        if padding != 0:
            phase = phase_padded[padding:-padding,padding:-padding]
        else:
            phase = phase_padded

        return phase

    def _generate_grid_FFT_from_file(
        self,
        filepath: str
    ):  
        
        filename, file_extension = os.path.splitext(filepath)

        if file_extension == '.txt':
            mask = np.loadtxt(filepath)

            if not mask.shape[0] == mask.shape[1]:
                raise Exception('Input txt mask is not isometric!')

        elif file_extension == '.npy':
            self.mask = np.load(filepath)

            if not all([N == self.mask.shape[0] for N in self.mask.shape]) and len(self.mask.shape) != 2:
                raise Exception('Input .npy mask is not isometric!')

        elif file_extension == '.mat':
            mask_dict = io.loadmat(filepath, mdict=None, appendmat=True)
        
            for key in mask_dict.keys():
                if isinstance(mask_dict[key], np.ndarray):
                    self.mask = mask_dict[key]
                    break

            if not all([N == self.mask.shape[0] for N in self.mask.shape]) and len(self.mask.shape) != 2:
                raise Exception('Input MATLAB mask is not isometric!')
        
        else:
            raise Exception('File did not have a supported file type (i.e.: .txt, .npy, .mat)')

        if self.mask.shape[0] != self.N:
            raise Exception('Input mask \'N\' is not the same as the grid \'N\'!')

        grid_dchi = np.array(self.mask, dtype=float)

        for i, dchi in enumerate(self.dchi_list):
            grid_dchi[grid_dchi == i+1] = dchi
        
        self.phase = self.dchi_mask_to_phase_FFT(grid_dchi, self.N, self.padding, self.dt)

    def _generate_grid_FFT_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel2D,
        extend: bool,
        progressbar: bool
    ):
        self.size = voxel.size

        N = self.N + 2*self.padding if extend else self.N

        self.mask = np.zeros((N,N), dtype=int)
        grid_dchi = np.zeros((N,N))

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            subvox_size = self.size/self.N
            is_IV = vsl.grid_is_IV(N, subvox_size) 

            mask_tmp = np.logical_and(np.logical_not(self.mask), is_IV)
            
            self.mask[mask_tmp] = vsl_counter + 1
            grid_dchi[mask_tmp] = vsl.dchi

        if extend:
            self.mask = self.mask[self.padding:-self.padding,self.padding:-self.padding]
            grid_dchi = np.roll(grid_dchi, shift=(-self.padding+1,-self.padding+1), axis=(0,1))
        
        self.permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]
        self.phase = self.dchi_mask_to_phase_FFT(grid_dchi, self.N, self.padding, self.B0, self.dt)

    def _generate_grid_analytical_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel2D,
        progressbar: bool
    ):
        self.size = voxel.size

        linear_coord = np.linspace(-self.size/2, self.size/2, self.N)
        X, Y = np.meshgrid(linear_coord, linear_coord)

        linear_pos = np.stack((X.ravel(), Y.ravel()), axis=1)
        linear_mask = np.zeros(self.N**2)
        linear_offset = np.zeros(self.N**2)

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):
            is_IV, dBz_EV, dBz_IV = vsl.dBz_mask_from_positions(linear_pos, self.B0)
            
            is_IV = np.logical_and(is_IV, (linear_mask == 0))
            linear_mask[is_IV] = vsl_counter + 1

            linear_offset += is_IV * dBz_IV + np.logical_not(is_IV) * dBz_EV

        self.mask = linear_mask.reshape(self.N,self.N)

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001
        self.phase = phase_conversion_factor * linear_offset.reshape(self.N,self.N)

        self.permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

    def _generate_mask_analytical_from_voxel(
        self,
        voxel: BOLDvoxel.Voxel2D,
        progressbar: bool
    ):
        self.mask = np.zeros((self.N,self.N))

        text = 'Creating Mask'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            subvox_size = self.size/self.N
            is_IV = vsl.grid_is_IV(self.N, subvox_size)

            mask_tmp = np.logical_and(np.logical_not(self.mask), is_IV)
            
            self.mask[mask_tmp] = vsl_counter + 1
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
        mask: np.ndarray,
        phase: np.ndarray,
        permeation_probability_list: List[float],
        size: float,
        B0: Optional[float]=None,
        dt: Optional[float]=None,
        padding: Optional[int]=None
    ):
        self.permeation_probability_list = permeation_probability_list
        self.mask = mask
        self.phase = phase
        self.size = size

        self.padding = padding
        self.B0 = B0
        self.dt = dt  

        N = self.mask.shape[0]

        if ((N, N, N) != self.mask.shape) or ((N, N, N) != self.mask.shape):
            raise Exception('Input \'mask\' or \'phase\' has inconsistent shape or is not isometric.')

        self.N = N

    @classmethod
    def from_voxel_analytical(
        cls,
        B0: float,
        dt: float,
        N: int,
        voxel: BOLDvoxel.Voxel3D,
        progressbar: bool
    ):
        size = voxel.size

        linear_coord = np.linspace(-size/2, size/2, N)
        X, Y, Z = np.meshgrid(linear_coord, linear_coord, linear_coord)

        linear_pos = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        linear_mask = np.zeros(N**3)
        linear_offset = np.zeros(N**3)

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            is_IV, dBz_EV, dBz_IV = vsl.dBz_mask_from_positions(linear_pos, B0)
            
            is_IV = np.logical_and(is_IV, (linear_mask == 0))
            linear_mask[is_IV] = vsl_counter + 1

            linear_offset += is_IV * dBz_IV + np.logical_not(is_IV) * dBz_EV

        mask = linear_mask.reshape(N, N, N)

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * dt * 0.001
        phase = phase_conversion_factor * linear_offset.reshape(N, N, N)

        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        return cls(
            mask=mask,
            phase=phase,
            permeation_probability_list=permeation_probability_list,
            size=size,
            B0=B0,
            dt=dt
        )

    @classmethod
    def from_voxel_FFT(
        cls,
        B0: float,
        dt: float,
        N: int,
        voxel: BOLDvoxel.Voxel3D,
        extend: bool=False,
        padding: int=0,
        progressbar: bool=True
    ):
        size = voxel.size

        N = N + 2*padding if extend else N

        mask = np.zeros((N,N,N), dtype=int)
        grid_dchi = np.zeros((N,N,N))

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):

            subvox_size = size/N
            is_IV = vsl.grid_is_IV(N, subvox_size)

            mask_tmp = np.logical_and(np.logical_not(mask), is_IV)
            
            mask[mask_tmp] = vsl_counter + 1
            grid_dchi[mask_tmp] = vsl.dchi
        
        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        if extend:
            phase = cls.dchi_mask_to_phase_FFT(grid_dchi, 0, dt, B0)
        else:
            phase = cls.dchi_mask_to_phase_FFT(grid_dchi, padding, dt, B0)

        if extend and padding > 0:
            mask = mask[padding:-padding,padding:-padding,padding:-padding]
            phase = phase[padding:-padding,padding:-padding,padding:-padding]

        return cls(
            mask=mask,
            phase=phase,
            permeation_probability_list=permeation_probability_list,
            size=size,
            B0=B0,
            dt=dt,
            padding=padding
        )
    
    @classmethod
    def from_file_FFT(
        cls,
        filepath: str,
        dchi_list: List[float],
        permeation_probability_list: List[float],
        size: float,
        dt: float,
        B0: float,
        padding: int=0
    ):  
        
        filename, file_extension = os.path.splitext(filepath)

        if file_extension == '.txt':
            mask = np.loadtxt(filepath)

            if not mask.shape[0]**2 == mask.shape[1]:
                raise Exception('Input txt mask is not isometric!')

            mask = mask.reshape((mask.shape[0], mask.shape[0], mask.shape[0]))

        elif file_extension == '.npy':
            mask = np.load(filepath)

        elif file_extension == '.mat':
            mask_dict = io.loadmat(filepath, mdict=None, appendmat=True)
        
            mask = mask_dict['mask']
        
        else:
            raise Exception('File did not have a supported file type (i.e.: .txt, .npy, .mat)')

        grid_dchi = np.array(mask, dtype=float)

        for i, dchi in enumerate(dchi_list):
            grid_dchi[grid_dchi == i+1] = dchi
        
        phase = cls.dchi_mask_to_phase_FFT(grid_dchi, padding, dt, B0)

        return cls(
            mask=mask,
            phase=phase,
            permeation_probability_list=permeation_probability_list,
            size=size,
            B0=B0,
            dt=dt,
            padding=padding
        )

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
        
        if padding > 0:
            phase = phase_padded[padding:-padding,padding:-padding,padding:-padding]
        else:
            phase = phase_padded

        return phase

class Grid2D:

    def __init__(
        self,
        mask: np.ndarray,
        phase: np.ndarray,
        permeation_probability_list: List[float],
        size: float,
        B0: Optional[float]=None,
        dt: Optional[float]=None,
        padding: Optional[int]=None
    ):
        self.permeation_probability_list = permeation_probability_list
        self.mask = mask
        self.phase = phase
        self.size = size

        self.padding = padding
        self.B0 = B0
        self.dt = dt

        N = self.mask.shape[0]

        if ((N, N) != self.mask.shape) or ((N, N) != self.mask.shape):
            raise Exception('Input \'mask\' or \'phase\' has inconsistent shape or is not isometric.')

        self.N = N

    @classmethod
    def fromVoxelAnalytical(
        cls,
        B0: float,
        dt: float,
        N: int,
        voxel: BOLDvoxel.Voxel2D,
        progressbar: bool
    ):
        size = voxel.size

        linear_coord = np.linspace(-size/2, size/2, N)
        X, Y = np.meshgrid(linear_coord, linear_coord)

        linear_pos = np.stack((X.ravel(), Y.ravel()), axis=1)
        linear_mask = np.zeros(N**2)
        linear_offset = np.zeros(N**2)

        text = 'Creating Offset Grid'
        for vsl_counter, vsl in tqdm(enumerate(voxel.vessels), total=len(voxel.vessels), desc=text, disable=not progressbar):
            is_IV, dBz_EV, dBz_IV = vsl.dBz_mask_from_positions(linear_pos, B0)
            
            is_IV = np.logical_and(is_IV, (linear_mask == 0))
            linear_mask[is_IV] = vsl_counter + 1

            linear_offset += is_IV * dBz_IV + np.logical_not(is_IV) * dBz_EV

        mask = linear_mask.reshape(N, N)

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * dt * 0.001
        phase = phase_conversion_factor * linear_offset.reshape(N, N)

        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        return cls(
            mask=mask,
            phase=phase,
            permeation_probability_list=permeation_probability_list,
            size=size,
            B0=B0,
            dt=dt
        )
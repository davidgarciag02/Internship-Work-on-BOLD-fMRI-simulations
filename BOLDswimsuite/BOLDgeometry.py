from __future__ import annotations
import numpy as np
import os
from scipy import fft, io
from typing import  List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from . import BOLDvessel

def size_from_k(diameter: float, k: float, ADC: float, dt: float) -> float:
    
    A = np.sqrt(2 * ADC * dt / 1000)
   
    return 2 * (A + k * diameter / 2)

class Geometry:
    
    def __init__(self, ndims: int):
        self._ndims = ndims

    def vessel_indices_from_positions(self, positions: np.ndarray):
        pass
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray):
        pass
    
    def wrap_boundary_positions(self, positions: np.ndarray):
        pass
    
    def permeates(self, vessel_index: int, random_float: float):
        pass

    def get_CBV(self):
        pass

    def _validate_positions(self, positions: np.ndarray):
        if len(positions.shape) == 2 and positions.shape[1] == self._ndims and issubclass(positions.dtype.type, np.floating):
            return
        raise Exception(f'\'positions\' must be a Numpy floating point array or shape (N, {self._ndims})')

class ContinuousVoxel(Geometry):
    
    def __init__(
        self,
        ndims: int,
        size: float,
        B0: float,
        vessels: Optional[Union[List[BOLDvessel.Vessel3D], List[BOLDvessel.Vessel2D]]]=None
    ):
        self.vessels = vessels if vessels is not None else []
        self.size: float = size
        self.B0 = B0
        self._ndims = ndims
    
    def vessel_indices_from_positions(self, positions: np.ndarray):
        self._validate_positions(positions)

        num_positions = positions.shape[0]

        # checks if new spin positions are IV and calculates dBz and phase
        vessel_indices = np.zeros(num_positions, dtype=int)

        for i, vessel in enumerate(self.vessels):
            is_IV = vessel.is_IV(positions)
            is_IV = np.logical_and(is_IV, np.logical_not(vessel_indices))

            vessel_indices[is_IV] = i+1

        return vessel_indices
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray):
        self._validate_positions(positions)

        num_positions = positions.shape[0]

        # checks if new spin positions are IV and calculates dBz and phase
        vessel_indices = np.zeros(num_positions, dtype=int)
        dBz = np.zeros(num_positions)

        for i, vessel in enumerate(self.vessels):
            is_IV, dBz_EV, dBz_IV = vessel.is_IV_dBz(positions, self.B0)
            is_IV = np.logical_and(is_IV, np.logical_not(vessel_indices))

            vessel_indices[is_IV] = i+1
            dBz += np.logical_not(is_IV) * dBz_EV + is_IV * dBz_IV

        return dBz, vessel_indices
    
    def wrap_boundary_positions(self, positions: np.ndarray):
        self._validate_positions(positions)
        
        positions += self.size * \
            (1 * (positions < -0.5 * self.size) - \
                1 * (positions > 0.5 * self.size))
        
        return positions
    
    def permeates(self, vessel_index: int, random_float: float):
        permeation_probability = self.vessels[vessel_index-1].permeation_probability
        permeates = random_float < permeation_probability
        return permeates
    
    def get_CBV(self):
        CBV = 0

        for vsl in self.vessels:
            CBV += vsl.volume_fraction(self.size)

        return CBV


class ContinuousVoxel3D(ContinuousVoxel):
     
    def __init__(
        self,
        size: float,
        B0: float,
        vessels: Optional[BOLDvessel.Vessel3D]=None
    ):  
        super().__init__(
            ndims=3,
            size=size,
            B0=B0,
            vessels=vessels
        )

    def add_vessel(
        self,
        vessel: BOLDvessel.Vessel3D
    ) -> ContinuousVoxel3D:

        self.vessels.append(vessel)
        return self
    
    @classmethod
    def from_random(
        cls,
        size: float,
        CBV: float,
        B0: float,
        labels: List[str],
        weights: Dict[str, float],
        diameter_distributions: Dict[str, List[float]],
        dchis: Dict[str, float],
        permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> ContinuousVoxel:

        rng = np.random.default_rng(seed)

        voxel = cls(
            size=size,
            B0=B0
        )

        str2vessel_class = {
            'cylinder': BOLDvessel.InfiniteCylinder3D,
            'sphere': BOLDvessel.Sphere3D
        }

        vessel_class = str2vessel_class[vessel_type]

        total_CBV = 0  # initializing target CBV
        current_CBV = 0  # initializing current CBV
        
        total_weight = 0 # initializing total CBV weight
        for label in labels:
            total_weight += weights[label]

        text = 'Populating Voxel'
        with tqdm(total=100, desc=text, disable=not progressbar) as pbar:

            # iterating through all vessel types
            for label in labels:
                # CBV occupied by the current vessel type
                type_CBV = CBV * weights[label] / total_weight
                # target CBV incremented by the vessel type's CBV
                total_CBV += type_CBV

                # generate vessels until the target CBV is reached
                while current_CBV < total_CBV:

                    vessel_intersects = True

                    counter = 0
                    while vessel_intersects:
                        # picks diameter
                        diameters = diameter_distributions[label]
                        diameter = rng.choice(diameters)

                        # picks dChi
                        dchi = dchis[label]

                        # picks permeation probability
                        permeation_probability = permeation_probabilities[label] if permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel: BOLDvessel.Vessel3D = vessel_class.from_random(
                            diameter=diameter,
                            dchi=dchi,
                            voxel_size=voxel.size,
                            permeation_probability=permeation_probability,
                            label=label,
                            rng=rng
                        )                        
                        
                        #check that vessel does not intersect other vessels
                        vessel_intersects = False
                        
                        if not allow_vessel_intersection:
                            for vsl in voxel.vessels:
                                if vsl.intersects(vessel):
                                    vessel_intersects = True
                                    counter += 1
                                    break

                    # adding the vessel to the list
                    voxel.add_vessel(vessel)
                    
                    # adding volume contribution from new vessel
                    current_CBV += vessel.volume_fraction(voxel.size)
                    
                    #manual override of the progress bar
                    progress_percentage = int(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()

        return voxel

class ContinuousVoxel2D(ContinuousVoxel):
    
    def __init__(
        self,
        size: float,
        B0: float,
        vessels: Optional[BOLDvessel.Vessel2D]=None
    ):  
        super().__init__(
            ndims=2,
            size=size,
            B0=B0,
            vessels=vessels
        )
    
    def add_vessel(
        self,
        vessel: BOLDvessel.Vessel2D
    ) -> ContinuousVoxel2D:

        self.vessels.append(vessel)
        return self

    @classmethod
    def from_random(
        cls,
        size: float,
        CBV: float,
        B0: float,
        labels: List[str],
        weights: Dict[str, float],
        diameters_distributions: Dict[str, List[float]],
        dchis: Dict[str, float],
        permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> ContinuousVoxel2D:

        rng = np.random.default_rng(seed)

        voxel = cls(
            size=size,
            B0=B0
        )

        str2vessel_class = {
            'cylinder': BOLDvessel.InfiniteCylinder2D,
        }

        vessel_class = str2vessel_class[vessel_type]

        total_CBV = 0  # initializing target CBV
        current_CBV = 0  # initializing current CBV
        
        total_weight = 0
        for label in labels:
            total_weight += weights[label]

        text = 'Populating Voxel'
        with tqdm(total=100, desc=text, disable=not progressbar) as pbar:
             # iterating through all vessel types
            for label in labels:
                # CBV occupied by the current vessel type
                type_CBV = CBV * weights[label] / total_weight
                # target CBV incremented by the vessel type's CBV
                total_CBV += type_CBV

                # generate vessels until the target CBV is reached
                while current_CBV < total_CBV:

                    vessel_intersects = True

                    counter = 0
                    while vessel_intersects:
                        # picks diameter
                        diameters = diameters_distributions[label]
                        diameter = rng.choice(diameters)

                        # picks dChi
                        dchi = dchis[label]

                        # picks permeation probability (impermeable if not set)
                        permeation_probability = permeation_probabilities[label] if permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel: BOLDvessel.Vessel2D = vessel_class.from_random(
                            diameter=diameter,
                            dchi=dchi,
                            voxel_size=voxel.size,
                            permeation_probability=permeation_probability,
                            label=label,
                            rng=rng
                        )                        
                        
                        #check that vessel does not intersect other vessels
                        vessel_intersects = False
                        
                        if not allow_vessel_intersection:
                            for vessel in voxel.vessels:
                                if vessel.intersects(vessel):
                                    vessel_intersects = True
                                    counter += 1
                                    break

                    # adding the vessel to the list
                    voxel.add_vessel(vessel)
                    
                    # adding volume contribution from new vessel
                    current_CBV += vessel.volume_fraction(voxel.size)
                    
                    #manual override of the progress bar
                    progress_percentage = round(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()
        
        return voxel

class DiscreteVoxel(Geometry):

    def __init__(
        self,
        ndims: int,
        vessel_index_grid: np.ndarray,
        dBz_grid: np.ndarray,
        permeation_probability_list: List[float],
        size: float
    ):
        if not np.issubdtype(vessel_index_grid.dtype, np.integer):
            raise Exception('Input \'vessel_index_mask\' is not of integer type.')
        
        self._ndims = ndims
        self.permeation_probability_list = permeation_probability_list
        self.vessel_index_grid = vessel_index_grid
        self.dBz_grid = dBz_grid
        self.size = size

        N = self.vessel_index_grid.shape[0]

        if ((N,)*self._ndims != self.vessel_index_grid.shape) or ((N,)*self._ndims != self.dBz_grid.shape):
            raise Exception('Input \'vessel_index_mask\' or \'dBz\' has inconsistent shape or is not isometric.')

        self.N = N

    def _position_to_grid(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        subvoxel_per_mm = self.N / self.size
        grid_positions = ((positions + 0.5 * self.size ) * subvoxel_per_mm).astype(int)
        grid_position_tuple = tuple((grid_positions[:, i] for i in range(self._ndims)))

        return grid_position_tuple
    
    def vessel_indices_from_positions(self, positions: np.ndarray):
        self._validate_positions(positions)

        grid_positions = self._position_to_grid(positions)
        vessel_indices = self.vessel_index_grid[grid_positions]

        return vessel_indices
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray):
        self._validate_positions(positions)

        grid_positions = self._position_to_grid(positions)
        dBz = self.dBz_grid[grid_positions]
        vessel_indices = self.vessel_index_grid[grid_positions]

        return dBz, vessel_indices
    
    def wrap_boundary_positions(self, positions: np.ndarray):
        self._validate_positions(positions)

        positions += self.size * \
            (1 * (positions < -0.5 * self.size) - \
                1 * (positions > 0.5 * self.size))
        
        return positions
    
    def permeates(self, vessel_index: int, random_float: float):
        permeation_probability = self.permeation_probability_list[vessel_index-1]
        permeates = random_float < permeation_probability
        return permeates
    
    def get_CBV(self):
        CBV = np.count_nonzero(self.vessel_index_grid)/self.vessel_index_grid.size

        return CBV

class DiscreteVoxel3D(DiscreteVoxel):

    def __init__(
        self,
        vessel_index_grid: np.ndarray,
        dBz_grid: np.ndarray,
        permeation_probability_list: List[float],
        size: float
    ):
        super().__init__(
            ndims=3,
            vessel_index_grid=vessel_index_grid,
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )

    @classmethod
    def from_continuous_analytical(
        cls,
        N: int,
        voxel: ContinuousVoxel3D
    ):
        size = voxel.size

        linear_coord = np.linspace(-size/2, size/2, N)
        X, Y, Z = np.meshgrid(linear_coord, linear_coord, linear_coord, indexing='ij')

        linear_positions = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

        linear_dBz, linear_vessel_indices = voxel.dBz_vessel_indices_from_positions(linear_positions)

        vessel_index_grid = linear_vessel_indices.reshape(N, N, N)
        dBz_grid = linear_dBz.reshape(N, N, N)

        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        return cls(
            vessel_index_grid=vessel_index_grid,
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )

    @classmethod
    def from_continuous_FFT(
        cls,
        N: int,
        voxel: ContinuousVoxel,
        padding: int=0,
        extend: bool=False,
    ):
        size = voxel.size
        subvox_size = size/N

        N = N + 2*padding if extend else N

        vessel_index_grid = np.zeros((N,N,N), dtype=int)
        grid_dchi = np.zeros((N,N,N))

        for vsl_counter, vsl in enumerate(voxel.vessels):
            
            is_IV = vsl.grid_is_IV(N, subvox_size)

            mask_tmp = np.logical_and(np.logical_not(vessel_index_grid), is_IV)
            
            vessel_index_grid[mask_tmp] = vsl_counter + 1
            grid_dchi[mask_tmp] = vsl.dchi
        
        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        if extend:
            dBz_grid = cls.dchi_mask_to_dBz_FFT(grid_dchi, 0, voxel.B0)
        else:
            dBz_grid = cls.dchi_mask_to_dBz_FFT(grid_dchi, padding, voxel.B0)

        if extend and padding > 0:
            vessel_index_grid = vessel_index_grid[padding:-padding,padding:-padding,padding:-padding]
            dBz_grid = dBz_grid[padding:-padding,padding:-padding,padding:-padding]

        return cls(
            vessel_index_grid=vessel_index_grid,
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )
    
    @classmethod
    def from_file_FFT(
        cls,
        filepath: str,
        dchi_list: List[float],
        permeation_probability_list: List[float],
        size: float,
        B0: float,
        padding: int=0
    ):  
        
        filename, file_extension = os.path.splitext(filepath)

        if file_extension == '.txt':
            vessel_index_grid = np.loadtxt(filepath)

            if not vessel_index_grid.shape[0]**2 == vessel_index_grid.shape[1]:
                raise Exception('Input txt mask is not isometric!')

            vessel_index_grid = vessel_index_grid.reshape((vessel_index_grid.shape[0], vessel_index_grid.shape[0], vessel_index_grid.shape[0]))

        elif file_extension == '.npy':
            vessel_index_grid = np.load(filepath)

        elif file_extension == '.mat':
            mask_dict = io.loadmat(filepath, mdict=None, appendmat=True)
        
            vessel_index_grid = mask_dict['mask']
        
        else:
            raise Exception('File did not have a supported file type (i.e.: .txt, .npy, .mat)')

        grid_dchi = np.array(vessel_index_grid, dtype=float)

        for i, dchi in enumerate(dchi_list):
            grid_dchi[grid_dchi == i+1] = dchi
        
        dBz_grid = cls.dchi_mask_to_dBz_FFT(grid_dchi, padding, B0)

        return cls(
            vessel_index_grid=vessel_index_grid.astype(int),
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )
    
    @classmethod
    def from_vessel_index_grid_FFT(
        cls,
        vessel_index_grid: np.ndarray,
        dchi_list: List[float],
        permeation_probability_list: List[float],
        size: float,
        B0: float,
        padding: int=0
    ):

        grid_dchi = np.array(vessel_index_grid, dtype=float)

        for i, dchi in enumerate(dchi_list):
            grid_dchi[grid_dchi == i+1] = dchi
        
        dBz_grid = cls.dchi_mask_to_dBz_FFT(grid_dchi, padding, B0)

        return cls(
            vessel_index_grid=vessel_index_grid.astype(int),
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )

    @staticmethod
    def dchi_mask_to_dBz_FFT(
        grid_dchi: np.ndarray,
        padding: int,
        B0: float
    ):
        N = grid_dchi.shape[0]
        half_N = int(np.ceil(N/2))

        if N % 2 != 0:
            pos_range = np.linspace(-half_N + 1, half_N - 1, N) 
        else:
            pos_range = np.linspace(-half_N + 1, half_N, N)
        
        X, Y, Z = np.meshgrid(pos_range, pos_range, pos_range, indexing='ij')
        r_squared = X**2 + Y**2 + Z**2

        zeros_indices = np.where(r_squared == 0)
        r_squared[zeros_indices] = 1 #dummy value to prevent divide by 0
        kernel_pos = (1/(4*np.pi)) * (3*Z**2 - r_squared) / (r_squared**(5/2))

        kernel_pos[zeros_indices] = 0 #set divide by 0 value to 0

        pad = padding * 2 + N

        kernel_FFT = fft.rfftn(kernel_pos, s=(pad,pad,pad))

        susceptibility_map_FFT = fft.rfftn(4*np.pi*grid_dchi, s=(pad,pad,pad))
        
        dBz_padded = B0 * fft.irfftn(
            susceptibility_map_FFT * kernel_FFT, 
            s=(pad, pad, pad), 
            overwrite_x = True
        )

        shift = int(np.floor(N/2) - 1) - padding
        dBz_padded = np.roll(dBz_padded, shift=(-shift, -shift, -shift), axis=(0,1,2))
        
        if padding > 0:
            dBz = dBz_padded[padding:-padding,padding:-padding,padding:-padding]
        else:
            dBz = dBz_padded

        return dBz

class DiscreteVoxel2D(DiscreteVoxel):

    def __init__(
        self,
        vessel_index_grid: np.ndarray,
        dBz_grid: np.ndarray,
        permeation_probability_list: List[float],
        size: float
    ):
        super().__init__(
            ndims=2,
            vessel_index_grid=vessel_index_grid,
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )

    @classmethod
    def from_continuous_analytical(
        cls,
        N: int,
        voxel: ContinuousVoxel2D
    ):
        size = voxel.size

        linear_coord = np.linspace(-size/2, size/2, N)
        X, Y = np.meshgrid(linear_coord, linear_coord, indexing='ij')

        linear_positions = np.stack((X.ravel(), Y.ravel()), axis=1)

        linear_dBz, linear_vessel_indices = voxel.dBz_vessel_indices_from_positions(linear_positions)

        vessel_index_grid = linear_vessel_indices.reshape(N, N)
        dBz_grid = linear_dBz.reshape(N, N)

        permeation_probability_list = [vsl.permeation_probability for vsl in voxel.vessels]

        return cls(
            vessel_index_grid=vessel_index_grid,
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )
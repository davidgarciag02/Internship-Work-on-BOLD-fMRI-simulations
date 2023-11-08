from __future__ import annotations
import numpy as np
import os
from scipy import fft, io
from typing import  List, Dict, Optional, Tuple, Union, Literal
from tqdm import tqdm
from . import BOLDvessel

def size_from_k(diameter: float, k: float, ADC: float, dt: float) -> float:
    
    A = np.sqrt(2 * ADC * dt / 1000)
    
    return 2 * (A + k * diameter / 2)

class Geometry:
    
    def __init__(self, ndims: int):
        self._ndims = ndims

    def vessel_indices_from_positions(self, positions: np.ndarray) -> np.ndarray:
        """Given an array of positions, returns the vessel index of each position.

        Parameters
        ----------
        positions : np.ndarray
            Positons in cartesian space (mm). Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Returns
        -------
        np.ndarray
            Array with the vessel index of each position.
        """        
        pass
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given an array of positions, returns both the dBz magnetic field offset and the vessel index of each position.

        Parameters
        ----------
        positions : np.ndarray
            Positons in cartesian space (mm). Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            2 element Tuple. The first element is an array with the dBz magnetic field offset of each position. The second element is an array with the vessel index of each position.
        """     
        pass
    
    def wrap_boundary_positions(self, positions: np.ndarray) -> np.ndarray:
        """Given an array of position, returns an array of positions where any positions outside the boundaries of the voxel are wrapped to the other side. Any positions that are not out-of-bounds are left untouched.

        Parameters
        ----------
        positions : np.ndarray
            Positons in cartesian space (mm). Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Returns
        -------
        np.ndarray
            Array of wrapped positions.
        """        
        pass
    
    def permeates(self, vessel_index: int, random_sample: Optional[float] = None) -> bool:
        """Given a vessel index, returns whether the vessel corresponding to the vessel index is permeated.

        Parameters
        ----------
        vessel_index : int
            Vessel index of the desired vessel.
        random_sample : Optional[float], optional
            A float within the interval [0.0, 1.0). By default, a valid random number will be generated.

        Returns
        -------
        bool
            Whether the vessel is permeated.
        """    
        pass

    def get_CBV(self) -> float:
        """Returns the estimated Cerebral Blood Volume, or the volume fraction of vessels within the voxel.

        Returns
        -------
        float
            Estimated Cerebral Blood Volume.
        """        
        pass

    def _validate_positions(self, positions: np.ndarray) -> None:
        """Validates the `position` argument for other functions in `Geometry`.

        Parameters
        ----------
        positions : np.ndarray
            Positons in cartesian space (mm). Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Raises
        ------
        Exception
            Occurs when an invalid `positions` argument has been provided.
        """  

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
    
    def vessel_indices_from_positions(self, positions: np.ndarray) -> np.ndarray:   
        self._validate_positions(positions)

        num_positions = positions.shape[0]

        # checks if new spin positions are IV and calculates dBz and phase
        vessel_indices = np.zeros(num_positions, dtype=int)

        for i, vessel in enumerate(self.vessels):
            is_IV = vessel.is_IV(positions)
            is_IV = np.logical_and(is_IV, np.logical_not(vessel_indices))

            vessel_indices[is_IV] = i+1

        return vessel_indices
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def wrap_boundary_positions(self, positions: np.ndarray) -> np.ndarray:
        self._validate_positions(positions)
        
        positions += self.size * \
            (1 * (positions < -0.5 * self.size) - \
                1 * (positions > 0.5 * self.size))
        
        return positions
    
    def permeates(self, vessel_index: int, random_sample: Optional[float] = None) -> bool:
        
        if random_sample is None: random_sample = np.random.random()

        permeation_probability = self.vessels[vessel_index-1].permeation_probability
        permeates = random_sample < permeation_probability
        return permeates
    
    def get_CBV(self) -> float:
        CBV = 0

        for vsl in self.vessels:
            CBV += vsl.volume_fraction(self.size)

        return CBV

class ContinuousVoxel3D(ContinuousVoxel):
    """Continuous space 3 dimensional voxel.

    Parameters
    ----------
    size : float
        Side length of the voxel (mm). Voxels are isometric.
    B0 : float
        B0 magnetic field strength (Tesla).
    vessels : Optional[BOLDvessel.Vessel3D], optional
        List of 3D vessel objects. By default will create an empty voxel, to which vessels can be added using the `add_vessel` method.
    """   

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
        """Add a 3D vessel to the voxel.

        Parameters
        ----------
        vessel : BOLDvessel.Vessel3D
            3D vessel object to add to the voxel.

        Returns
        -------
        ContinuousVoxel3D
            The voxel with the vessel added to it.
        """

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
        vessel_type: Literal['cylinder', 'sphere']='cylinder',
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> ContinuousVoxel3D:
        """Alternate constructor, randomly generates the 3D continuous voxel given a set of parameters.

        Parameters
        ----------
        size : float
            Side length of the voxel (mm). Voxels are isometric.
        CBV : float
            Target cerebral blood volume (CBV). Vessels will be added until the estimated CBV reaches the target CBV (the vessel that causes the estimated CBV to surpass the target CBV is included).
        B0 : float
            B0 magnetic field strength (Tesla). The field direction is always [0, 0, 1].
        labels : List[str]
            List of vessel group labels. 
        weights : Dict[str, float]
            Dictionary with each group label (keys) and a relative CBV weight of each group (values). The sum of CBV weight does not need to equal 1, as the weighting is normalized. 
        diameter_distributions : Dict[str, List[float]]
            Dictionary with each group label (keys) and a list of diameters of each group (values). The generated vessel diameters are uniformly sampled from the list.
        dchis : Dict[str, float]
            Dictionary with each group label (keys) and the magnetic susceptibility difference of each group (values). The magnetic susceptibility difference is between the vessel's intravascular compartment and the extravascular space. The magnetic susceptibility units are in cgs.
        permeation_probabilities : Optional[Dict[str, float]], optional
            Dictionary with each group label (keys) and the permeation probability of each group (values). The permeation probability applies only to Monte Carlo simulations. Any spins diffusing across the vessel wall during a Monte Carlo step will have this probability of permeating through. The default value will set all probabilities to 0, making the vessels impermeable.
        vessel_type : str, optional
            Type of vessel to generate. 'cylinder' will generate `BOLDvessel.InfiniteCylinder3D` and 'sphere' will generate spheres `BOLDvessel.Sphere3D`. Default is 'cylinder'.
        allow_vessel_intersection : bool, optional
            When generating the vessels, whether to allow them to intersect. Setting to False in voxels with InfiniteCylinder3D vessels creates a voxel with non-uniform CBV. The default is True.
        seed : Optional[int], optional
            Seed for the random generation. The default is None, which does not use a seed.
        progressbar : bool, optional
            Whether to show a progress bar in the terminal, by default True.

        Returns
        -------
        ContinuousVoxel3D
            3D continuous voxel.
        """

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
    """Continuous space 2 dimensional voxel.

    Parameters
    ----------
    size : float
        Side length of the voxel (mm). Voxels are isometric.
    B0 : float
        B0 magnetic field strength (Tesla).
    vessels : Optional[BOLDvessel.Vessel2D], optional
        List of 3D vessel objects. By default will create an empty voxel, to which vessels can be added using the `add_vessel` method.
    """     

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
        """Add a 2D vessel to the voxel.

        Parameters
        ----------
        vessel : BOLDvessel.Vessel2D
            2D vessel object to add to the voxel.

        Returns
        -------
        ContinuousVoxel2D
            The voxel with the vessel added to it.
        """

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
        vessel_type: Literal['cylinder']='cylinder',
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> ContinuousVoxel2D:
        """Alternate constructor, randomly generates the 2D continuous voxel given a set of parameters.

        Parameters
        ----------
        size : float
            Side length of the voxel (mm). Voxels are isometric.
        CBV : float
            Target cerebral blood volume (CBV). Vessels will be added until the estimated CBV reaches the target CBV (the vessel that causes the estimated CBV to surpass the target CBV is included).
        B0 : float
            B0 magnetic field strength (Tesla).
        labels : List[str]
            List of vessel group labels. 
        weights : Dict[str, float]
            Dictionary with each group label (keys) and a relative CBV weight of each group (values). The sum of CBV weight does not need to equal 1, as the weighting is normalized. 
        diameter_distributions : Dict[str, List[float]]
            Dictionary with each group label (keys) and a list of diameters of each group (values). The generated vessel diameters are uniformly sampled from the list.
        dchis : Dict[str, float]
            Dictionary with each group label (keys) and the magnetic susceptibility difference of each group (values). The magnetic susceptibility difference is between the vessel's intravascular compartment and the extravascular space. The magnetic susceptibility units are in cgs.
        permeation_probabilities : Optional[Dict[str, float]], optional
            Dictionary with each group label (keys) and the permeation probability of each group (values). The permeation probability applies only to Monte Carlo simulations. Any spins diffusing across the vessel wall during a Monte Carlo step will have this probability of permeating through. The default value will set all probabilities to 0, making the vessels impermeable.
        vessel_type : str, optional
            Type of vessel to generate. 'cylinder' will generate `BOLDvessel.InfiniteCylinder2D`. Default is 'cylinder'.
        allow_vessel_intersection : bool, optional
            When generating the vessels, whether to allow them to intersect. Setting to False in voxels with InfiniteCylinder3D vessels creates a voxel with non-uniform CBV. The default is True.
        seed : Optional[int], optional
            Seed for the random generation. The default is None, which does not use a seed.
        progressbar : bool, optional
            Whether to show a progress bar in the terminal, by default True.

        Returns
        -------
        ContinuousVoxel2D
            2D continuous voxel.
        """

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
                        diameters = diameter_distributions[label]
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
    
    def vessel_indices_from_positions(self, positions: np.ndarray) -> np.ndarray:
        self._validate_positions(positions)

        grid_positions = self._position_to_grid(positions)
        vessel_indices = self.vessel_index_grid[grid_positions]

        return vessel_indices
    
    def dBz_vessel_indices_from_positions(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_positions(positions)

        grid_positions = self._position_to_grid(positions)
        dBz = self.dBz_grid[grid_positions]
        vessel_indices = self.vessel_index_grid[grid_positions]

        return dBz, vessel_indices
    
    def wrap_boundary_positions(self, positions: np.ndarray) -> np.ndarray:
        self._validate_positions(positions)

        positions += self.size * \
            (1 * (positions < -0.5 * self.size) - \
                1 * (positions > 0.5 * self.size))
        
        return positions
    
    def permeates(self, vessel_index: int, random_sample: Optional[float] = None) -> bool:
        
        if random_sample is None: random_sample = np.random.random()

        permeation_probability = self.permeation_probability_list[vessel_index-1]
        permeates = random_sample < permeation_probability
        return permeates
    
    def get_CBV(self) -> float:
        CBV = np.count_nonzero(self.vessel_index_grid)/self.vessel_index_grid.size

        return CBV

class DiscreteVoxel3D(DiscreteVoxel):
    """Discrete space 3 dimensional voxel.

    Parameters
    ----------
    vessel_index_grid : np.ndarray
        Integer array of shape (N, N, N). It serves as a discretized representation of the voxel space, indicating where the intravascular and extravascular spaces are located. A value of 0 represents the extravascular space and positive integers (1,2,3...) represent intravascular space. Different positive integers represent different vessels, or vessel types, which can be associated with different properties. The integer associated to a specific vessel is called its 'vessel index'.
    dBz_grid : np.ndarray
        Float array of shape (N, N, N). It represents the magnetic field offset space (in Tesla). This should be matched to the `vessel_index_grid`.
    permeation_probability_list : List[float]
        List of probabilities (between 0 and 1) which indicate the probability for Monte Carlo spins to permeate in and out of the vessels. The first item in the list corresponds to the permeation probability of all vessels with a vessel index of 1, the second item in the list corresponds to the permeation probability of all vessels with a vessel index of 2, and so on for any additional vessel index. The extravascular space does not have a permeation probability, so a vessel index of 0 does not have an associated permeation probability in the list.
    size : float
        The side length of the voxel (in mm). Voxels are isometric.
    """

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
    ) -> DiscreteVoxel3D:
        """Alternate constructor, converts a 3D continuous voxel to a 3D discrete voxel. The magnetic field offset (dBz) is calculated using the analytical equations from the 3D continuous voxel object.

        Parameters
        ----------
        N : int
            The number of discrete points along the voxel edges. Therefore the output 3D discrete voxel is represented on an (N, N, N) grid.
        voxel : ContinuousVoxel3D
            3D continuous voxel object to convert to discrete space.

        Returns
        -------
        DiscreteVoxel3D
            3D discrete voxel.
        """        
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
    ) -> DiscreteVoxel3D:
        """Alternate constructor, converts a 3D continuous voxel to a 3D discrete voxel. The magnetic field offset (dBz) is calculated using FFT convolution.

        Parameters
        ----------
        N : int
            The number of discrete points along the voxel edges. Therefore the output 3D discrete voxel is represented on an (N, N, N) grid.
        voxel : ContinuousVoxel
            3D continuous voxel object to convert to discrete space.
        padding : int, optional
            Amount of zero padding to add to each side of the voxel before the FFT step. The default is 0, which will cause wrapping of the field offset. Using a value of N/2 will completely remove the wrapping effect but is more computationally demanding.
        extend : bool, optional
            If True, will extend the vessels to the zero padding, but is more computationally demanding. Doing so creates a more accurate representation of the continuous voxel (e.g. infinite cylinders cannot be infinite in the discrete space, but extending the vessels will make them "more" infinite). Default is False.

        Returns
        -------
        DiscreteVoxel3D
            3D discrete voxel.
        """

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
    ) -> DiscreteVoxel3D:
             
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
        dchis: Union[List[float], np.ndarray],
        permeation_probability_list: List[float],
        size: float,
        B0: float,
        padding: int=0
    ) -> DiscreteVoxel3D:
        """Alternate constructor, creates a 3D discrete voxel from an imported vessel index grid and magnetic susceptibility mapping. The magnetic field offset (dBz) is calculated using FFT convolution.

        Parameters
        ----------
        vessel_index_grid : np.ndarray
            _description_
        dchis : Union[List[float], np.ndarray]
            The susceptibility difference between the intravascular and extravascular space. Can be a list of magnetic susceptibility differences where the first item in the list corresponds to the magnetic susceptibility difference of of all vessels with a vessel index of 1, the second item in the list corresponds to the magnetic susceptibility difference of all vessels with a vessel index of 2, and so on for any additional vessel index. Can also be a float array of shape (N, N, N), indicating the magnetic susceptibility difference at each point in space. Units are in cgs.
        permeation_probability_list : List[float]
            List of probabilities (between 0 and 1) which indicate the probability for Monte Carlo spins to permeate in and out of the vessels. The first item in the list corresponds to the permeation probability of all vessels with a vessel index of 1, the second item in the list corresponds to the permeation probability of all vessels with a vessel index of 2, and so on for any additional vessel index. The extravascular space does not have a permeation probability, so a vessel index of 0 does not have an associated permeation probability in the list.
        size : float
            Side length of the voxel (mm). Voxels are isometric.
        B0 : float
            B0 magnetic field strength (Tesla).
        padding : int, optional
            Amount of zero padding to add to each side of the voxel before the FFT step. The default is 0, which will cause wrapping of the field offset. Using a value of N/2 will completely remove the wrapping effect but is more computationally demanding.

        Returns
        -------
        DiscreteVoxel3D
            3D discrete voxel.   
        """

        if isinstance(dchis, list):
            grid_dchi = np.array(vessel_index_grid, dtype=float)

            for i, dchi in enumerate(dchis):
                grid_dchi[grid_dchi == i+1] = dchi

        elif isinstance(dchis, np.ndarray):
            grid_dchi = dchis

        else:
            raise Exception('`dchis` argument is neither a list nor a Numpy array.')
        
        dBz_grid = cls.dchi_mask_to_dBz_FFT(grid_dchi, padding, B0)

        return cls(
            vessel_index_grid=vessel_index_grid.astype(int),
            dBz_grid=dBz_grid,
            permeation_probability_list=permeation_probability_list,
            size=size
        )

    @staticmethod
    def dchi_mask_to_dBz_FFT(
        dchi_grid: np.ndarray,
        padding: int,
        B0: float
    ):
        N = dchi_grid.shape[0]
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

        susceptibility_map_FFT = fft.rfftn(4*np.pi*dchi_grid, s=(pad,pad,pad))
        
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
    """Discrete space 2 dimensional voxel.

    Parameters
    ----------
    vessel_index_grid : np.ndarray
        Integer array of shape (N, N). It serves as a discretized representation of the voxel space, indicating where the intravascular and extravascular spaces are located. A value of 0 represents the extravascular space and positive integers (1,2,3...) represent intravascular space. Different positive integers represent different vessels, or vessel types, which can be associated with different properties. The integer associated to a specific vessel is called its 'vessel index'.
    dBz_grid : np.ndarray
        Float array of shape (N, N). It represents the magnetic field offset space (in Tesla). This should be matched to the `vessel_index_grid`.
    permeation_probability_list : List[float]
        List of probabilities (between 0 and 1) which indicate the probability for Monte Carlo spins to permeate in and out of the vessels. The first item in the list corresponds to the permeation probability of all vessels with a vessel index of 1, the second item in the list corresponds to the permeation probability of all vessels with a vessel index of 2, and so on for any additional vessel index. The extravascular space does not have a permeation probability, so a vessel index of 0 does not have an associated permeation probability in the list.
    size : float
        The side length of the voxel (in mm). Voxels are isometric.
    """
    
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
    ) -> DiscreteVoxel2D:
        """Alternate constructor, converts a 2D continuous voxel to a 2D discrete voxel. The magnetic field offset (dBz) is calculated using the analytical equations from the 2D continuous voxel object.

        Parameters
        ----------
        N : int
            The number of discrete points along the voxel edges. Therefore the output 3D discrete voxel is represented on an (N, N) grid.
        voxel : ContinuousVoxel2D
            2D continuous voxel object to convert to discrete space.

        Returns
        -------
        DiscreteVoxel2D
            2D discrete voxel.
        """

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
import numpy as np
from . import BOLDgeometry
from .BOLDconstants import *
from typing import Tuple, Optional, Union

class Spins:
    """Object used for Monte Carlo diffusion.

    Parameters
    ----------
    ndims : int
        Number of spatial dimensions.
    ADC : float
        Apparent diffusion coefficient (mm^2/s).
    num_spins : int
        Number of spins to simulate.
    geometry : BOLDgeometry.Geometry
        Geometry in which the spins will diffuse.
    dt : float
        Time step length for the inital phase calculation (ms).
    IV : bool, optional
        If False, will only place spins in the extravascular space. The default is True.
    seed : Optional[int], optional
        Seed for placing the spins and performing Monte Carlo steps. The default is None, which does not use a seed.
    """   

    def __init__(self,
        ndims: int,
        ADC: float,
        num_spins: int,
        geometry: BOLDgeometry.Geometry,
        dt: float,
        IV: bool=True,
        seed: Optional[int]=None
    ):     
        self._ndims = ndims       
        self.ADC = ADC
        self.num_spins = num_spins
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.dt = dt
        self.geometry = geometry

        # populating the inital position of the spins (at t=0)
        self.positions = self._place_spins(IV)            
      
        dBz, vessel_indices = self.geometry.dBz_vessel_indices_from_positions(positions=self.positions)

        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * dt * 0.001
        self.phase = phase_conversion_factor * dBz
        self.vessel_indices = vessel_indices

    def step(self, dt: float):
        """Advance all Monte Carlo spins by one step.

        Parameters
        ----------
        dt : float
            Length of time step (ms).
        """

        self.dt = dt
        
        # setting the conversion factor for frequency to phase
        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001

        # calculate the diffusion length
        diffusion_length = np.sqrt(self.ADC * 2 * (self.dt / 1000))
        
        # calculate a new position for all spins (random walk)
        new_positions = self.geometry.wrap_boundary_positions(
            self.positions + diffusion_length * self.rng.normal(size=(self.num_spins, self._ndims))
        )

        # checks if new spin positions are IV and calculates dBz and phase
        dBz, new_vessel_indices = self.geometry.dBz_vessel_indices_from_positions(positions=new_positions)

        # finds all spins that pass through a vessel wall (diffusing)
        permeating_spins_indices = np.flatnonzero(self.vessel_indices != new_vessel_indices)

        # initialize some lists for patching incorrect diffusion
        # positions
        fix_spins_indices = []

        # iterates through all diffusing spins
        for index in permeating_spins_indices:
            # finds if the spin was IV before and after the diffusion
            # through the vessel wall
            previous_is_IV = self.vessel_indices[index] != 0

            # finds the vessel object whose wall the spin is crossing
            if previous_is_IV:
                vessel_index = self.vessel_indices[index]
            else:
                vessel_index = new_vessel_indices[index]

            # create a bool according to the probability (if True the
            # spin is allowed to diffuse through the vessel wall)
            permeation_allowed = self.geometry.permeates(vessel_index, self.rng.random())

            if permeation_allowed:
                continue

            fix_spins_indices.append(index)

        #convert list to array for convenience
        fix_spins_indices = np.array(fix_spins_indices, dtype=int)
        fix_vessel_indices = self.vessel_indices[fix_spins_indices]

        num_fix = fix_spins_indices.size

        #recalculate the phase of the incorrect spins
        while num_fix > 0:
            # checks if new spin positions are IV and calculates dBz and phase
            new_positions[fix_spins_indices, :] = self.geometry.wrap_boundary_positions(
                self.positions[fix_spins_indices, :] + diffusion_length * self.rng.normal(size=(num_fix, self._ndims))
            )
            
            fix_dBz, fix_new_vessel_indices = self.geometry.dBz_vessel_indices_from_positions(positions=new_positions[fix_spins_indices, :])

            dBz[fix_spins_indices] = fix_dBz
            new_vessel_indices[fix_spins_indices] = fix_new_vessel_indices

            fix_spins_indices = fix_spins_indices[fix_vessel_indices != fix_new_vessel_indices]
            fix_vessel_indices = fix_vessel_indices[fix_vessel_indices != fix_new_vessel_indices]
            
            num_fix = fix_spins_indices.size

        # new position becomes the previous position
        self.phase = dBz * phase_conversion_factor
        self.positions = new_positions
        self.vessel_indices = new_vessel_indices

    def get_phase_vessel_indices_dt(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get phase, vessel index and dt information.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            3 element Tuple. The first element is an array of phase offset (in radians) for each spin. The second element is an array of vessel indices, indicating what space each spin is in. The third element is the time step length of the current step.
        """        
        return self.phase, self.vessel_indices, self.dt

    def _place_spins(
        self,
        IV: bool
    ) -> np.ndarray:
        """Populate the spins within the geometry.

        Parameters
        ----------
        IV : bool
            If False, will only place spins in the extravascular space. Otherwise will place spins everywhere.

        Returns
        -------
        np.ndarray
            Array of spin positions.
        """        

        if IV:
            return self.geometry.size*(self.rng.random((self.num_spins, self._ndims)) - 0.5)

        positions_list = []
        num_IV_spins = self.num_spins
        
        while num_IV_spins != 0:
            new_positions = self.geometry.size * (self.rng.random((self.num_spins, self._ndims)) - 0.5)
            vessel_indices = self.geometry.vessel_indices_from_positions(new_positions)
            num_IV_spins = np.flatnonzero(vessel_indices).size
            positions_list.append(new_positions[np.logical_not(vessel_indices)])

        positions = np.vstack(positions_list)

        return positions

class Spins3D(Spins):
    """Object used for Monte Carlo diffusion in 3D.

    Parameters
    ----------
    ADC : float
        Apparent diffusion coefficient (mm^2/s).
    num_spins : int
        Number of spins to simulate.
    geometry : Union[BOLDgeometry.DiscreteVoxel3D, BOLDgeometry.ContinuousVoxel3D]
        3D geometry in which the spins will diffuse.
    dt : float
        Time step length for the inital phase calculation (ms).
    IV : bool, optional
        If False, will only place spins in the extravascular space. The default is True.
    seed : Optional[int], optional
        Seed for placing the spins and performing Monte Carlo steps. The default is None, which does not use a seed.
    """

    def __init__(
        self,
        ADC: float,
        num_spins: int,
        geometry: Union[BOLDgeometry.DiscreteVoxel3D, BOLDgeometry.ContinuousVoxel3D],
        dt: float,
        IV: bool=True,
        seed: Optional[int]=None
    ):
        super().__init__(
            ndims=3,         
            ADC=ADC,
            num_spins=num_spins,
            geometry=geometry,
            dt=dt,
            IV=IV,
            seed=seed
        )

class Spins2D(Spins):
    """Object used for Monte Carlo diffusion in 2D.

    Parameters
    ----------
    ADC : float
        Apparent diffusion coefficient (mm^2/s).
    num_spins : int
        Number of spins to simulate.
    geometry : Union[BOLDgeometry.DiscreteVoxel3D, BOLDgeometry.ContinuousVoxel3D]
        2D geometry in which the spins will diffuse.
    dt : float
        Time step length for the inital phase calculation (ms).
    IV : bool, optional
        If False, will only place spins in the extravascular space. The default is True.
    seed : Optional[int], optional
        Seed for placing the spins and performing Monte Carlo steps. The default is None, which does not use a seed.
    """

    def __init__(
        self,
        ADC: float,
        num_spins: int,
        geometry: Union[BOLDgeometry.DiscreteVoxel2D, BOLDgeometry.ContinuousVoxel2D],
        dt: float,
        IV: bool=True,
        seed: Optional[int]=None
    ):
        super().__init__(
            ndims=2,         
            ADC=ADC,
            num_spins=num_spins,
            geometry=geometry,
            dt=dt,
            IV=IV,
            seed=seed
        )
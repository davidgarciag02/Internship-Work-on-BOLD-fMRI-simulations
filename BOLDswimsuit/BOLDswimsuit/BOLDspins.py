import numpy as np
from tqdm import tqdm
from . import BOLDvessel, BOLDvoxel, BOLDgrid
from .BOLDconstants import *

class SpinsContinuous3D:

    def __init__(self,
        ADC: float,
        B0: float,
        num_spins: int,
        num_dt: int,
        dt: float,
        edge_width: float=0,
        seed: int=None
    ):
        
        self.ADC = ADC
        self.B0 = B0
        self.num_spins = num_spins
        self.num_dt = num_dt
        self.dt = dt
        self.edge_width = edge_width

        self.positions = np.zeros((self.num_spins, self.num_dt, 3))
        self.is_IV_vessel = None
        self.sample = None

        self.phase = np.zeros((self.num_spins, self.num_dt))
        self.is_IV = np.zeros((self.num_spins, self.num_dt), dtype=bool)
        self.Mx = np.zeros((self.num_spins, self.num_dt + 1))
        self.My = np.zeros((self.num_spins, self.num_dt + 1))
        self.Mz = np.zeros((self.num_spins, self.num_dt + 1))

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def random_walk(
        self,
        voxel: BOLDvoxel.Voxel3D,
        IV: bool=True,
        record_is_IV_vessel: bool=False,
        progressbar: bool=True
    ) -> None:
        if record_is_IV_vessel:
            self.is_IV_vessel = np.empty((self.num_spins, self.num_dt), dtype=object)

        # setting the conversion factor for frequency to phase
        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001

        # sets the function to check for vessel wall diffusion
        if IV:
            check_diffusion = self.check_diffusion_EVIV
        elif not IV:
            check_diffusion = self.check_diffusion_EV

        # populating the inital position of the spins (at t=0)
        self.place_spins(voxel, IV, progressbar)

        # calculate the diffusion distance
        h = np.sqrt(self.ADC * 2 * (self.dt / 1000))

        # randomly positions spins in the voxel
        previous_position = np.empty((self.num_spins, 3))
        previous_position[:, :] = self.positions[:, 0, :]

        # initialize the array which will contain information about what spins
        # are in which vessels (IV spins)
        previous_is_IV_vsl = np.full(self.num_spins, None, dtype=object)

        dBz = np.zeros(self.num_spins)
        # checks if initial spin positions are IV and calculates dBz / phase
        for vessel in voxel.vessels:
            is_IV, dBz_EV, dBz_IV = vessel.dBz_mask_from_positions(previous_position, self.B0)
            is_IV = np.logical_and(is_IV, (previous_is_IV_vsl == None))

            previous_is_IV_vsl[is_IV] = vessel
            dBz += np.logical_not(is_IV)*dBz_EV + is_IV*dBz_IV

        self.is_IV[:, 0] = previous_is_IV_vsl != None
        self.phase[:, 0] = phase_conversion_factor * dBz
        
        if record_is_IV_vessel:
            self.is_IV_vessel[:, 0] = previous_is_IV_vsl

        # iterating through all time steps
        text = 'Diffusing Spins'
        for j in tqdm(range(1, self.num_dt), desc=text, disable=not progressbar):

            # calculate a new position for all spins (random walk)
            new_position = previous_position + h * self.rng.normal(size=(self.num_spins, 3))
            new_position += voxel.size * \
                (1 * (new_position < -0.5 * voxel.size) -
                    1 * (new_position > 0.5 * voxel.size))

            # checks if new spin positions are IV and calculates dBz /
            # phase
            new_is_IV_vsl = np.full(self.num_spins, None, dtype=object)
            dBz = np.zeros(self.num_spins)

            for vessel in voxel.vessels:
                is_IV, dBz_EV, dBz_IV = vessel.dBz_mask_from_positions(previous_position, self.B0)
                is_IV = np.logical_and(is_IV, (new_is_IV_vsl == None))

                new_is_IV_vsl[is_IV] = vessel
                dBz += np.logical_not(is_IV) * dBz_EV + is_IV * dBz_IV


            self.phase[:, j] = phase_conversion_factor * dBz

            # finds all spins that pass through a vessel wall (diffusing)
            permeating_spins_indices = ((previous_is_IV_vsl != new_is_IV_vsl)).nonzero()[0]

            # initialize some lists for patching incorrect diffusion
            # positions
            fix_spins_indices = []
            fix_spins_positions = []
            fix_spins_is_IV_vsl = []

            # iterates through all diffusing spins
            for i in permeating_spins_indices:
                # finds if the spin was IV before and after the diffusion
                # through the vessel wall
                previous_is_IV = previous_is_IV_vsl[i] is not None

                # finds the vessel object whose wall the spin is crossing
                if previous_is_IV:
                    vessel = previous_is_IV_vsl[i]
                else:
                    vessel = new_is_IV_vsl[i]

                # create a bool according to the probability (if True the
                # spin is allowed to diffuse through the vessel wall)
                diffuses = self.rng.random() < vessel.permeation_probability

                # if the spin is not allowed to diffuse through the vessel
                # wall
                if not diffuses:
                    # calculate new position until it does not diffuse
                    # across a vessel wall
                    is_diffused = True
                    while is_diffused:
                        new_position[i, :] = previous_position[i, :] + h * self.rng.normal(size=(3))
                        new_position[i,:] += voxel.size * \
                            (1 * (new_position[i,:] < -0.5 * voxel.size) - \
                                1 * (new_position[i,:] > 0.5 * voxel.size))

                        is_diffused = check_diffusion(
                            voxel,
                            vessel,
                            new_position[i, :],
                            previous_is_IV
                        )

                    new_is_IV_vsl[i] = previous_is_IV_vsl[i]

                    fix_spins_indices.append(i)
                    fix_spins_positions.append(new_position[i, :])
                    fix_spins_is_IV_vsl.append(previous_is_IV_vsl[i])

            #convert list to array for convenience
            fix_spins_indices = np.array(fix_spins_indices)
            fix_spins_positions = np.array(fix_spins_positions)
            fix_spins_is_IV_vsl = np.array(fix_spins_is_IV_vsl)

            #recalculate the phase of the incorrect spins
            if fix_spins_indices.size > 0:
                dBz = self.dBz_fix(fix_spins_positions, fix_spins_is_IV_vsl, voxel)
                self.phase[fix_spins_indices,j] = \
                    phase_conversion_factor * dBz

            # assign the new positions
            self.is_IV[:, j] = new_is_IV_vsl != None
            self.positions[:, j, :] = new_position

            if record_is_IV_vessel:
                self.is_IV_vessel[:, j] = new_is_IV_vsl

            # new position becomes the previous position
            previous_position = new_position
            previous_is_IV_vsl = new_is_IV_vsl

    def dBz_fix(
        self,
        positions: np.ndarray,
        is_IV_vsl: np.ndarray,
        voxel: BOLDvoxel.Voxel3D
    ) -> np.ndarray:
        dBz = np.zeros(is_IV_vsl.shape)

        # iterating through all vessels
        for vsl in voxel.vessels:
            # calculate the EV contribution of the vessel
            dBz += (is_IV_vsl != vsl) * vsl.dBz_EV(positions, self.B0) + \
                (is_IV_vsl == vsl) * vsl.dBz_IV(self.B0)

        return dBz

    def check_diffusion_EVIV(
        self,
        voxel: BOLDvoxel.Voxel3D,
        vessel: BOLDvessel.Vessel3D,
        position: np.ndarray,
        previous_is_IV: bool
    ) -> bool:
        # check if the spin has diffused in the specific vessel only
        position = np.expand_dims(position, axis=0)
        is_permeated = vessel.is_IV(position) != previous_is_IV
        return(is_permeated)
        

    def check_diffusion_EV(
        self,
        voxel: BOLDvoxel.Voxel3D,
        vessel: BOLDvessel.Vessel3D,
        position: np.ndarray,
        previous_is_IV: bool
    ) -> bool:
        # check if the spin has diffused in any vessels
        position = np.expand_dims(position, axis=0)
        for vessel in voxel.vessels:
            is_permeated = vessel.is_IV(position) != previous_is_IV
            if is_permeated:  # if the spin has diffused across a vessel wall, try again
                break
        return(is_permeated)

    def place_spins(
        self,
        voxel: BOLDvoxel.Voxel3D,
        IV: bool,
        progressbar: bool=True
    ) -> None:

        if IV:
            # creating extra- and intra-vascular spins
            self.positions[:, 0, :] = voxel.size * \
                (self.rng.random((self.num_spins, 3)) - 0.5)

        elif not IV:
            position = voxel.size * (self.rng.random((self.num_spins, 3)) - 0.5)

            is_IV = np.zeros(self.num_spins)
            for vessel in voxel.vessels:
                is_IV += vessel.is_IV(position)

            IV_spins = is_IV.nonzero()[0]

            text = 'Creating Spins'
            for i in tqdm(IV_spins, desc=text, disable=not progressbar):
                is_IV_fix = True
                while is_IV_fix:
                    # generate position
                    initial_position = voxel.size * (self.rng.random((1, 3)) - 0.5)
                    # check if position is in any vessel
                    for vessel in voxel.vessels:
                        is_IV_fix = vessel.is_IV(initial_position)
                        if is_IV_fix:
                            break

                position[i, :] = initial_position

            self.positions[:, 0, :] = position

    def sample_region(
        self,
        voxel: BOLDvoxel.Voxel3D
    ) -> None:

        # calculates the sample region edge position (center of region at
        # 0,0,0)
        sample_size = (voxel.size - 2 * self.edge_width * voxel.size) / 2

        # create the boolean array
        self.sample = np.concatenate(
            (np.ones((self.num_spins, 1)), 1 * np.all(np.abs(self.positions) <= sample_size, 2)), 1)

class SpinsDiscrete3D:

    def __init__(
        self,
        ADC: float,
        B0: float,
        num_spins: int,
        num_dt: int,
        dt: float,
        edge_width: float,
        seed: int=None
    ):

        self.ADC = ADC
        self.B0 = B0
        self.num_spins = num_spins
        self.num_dt = num_dt
        self.dt = dt
        self.N = None
        self.edge_width = edge_width

        self.positions = np.zeros((self.num_spins, self.num_dt, 3))
        self.sample = None

        self.phase = np.zeros((self.num_spins, self.num_dt))
        self.is_IV = np.zeros((self.num_spins, self.num_dt), dtype=int)
        self.Mx = np.zeros((self.num_spins, self.num_dt + 1))
        self.My = np.zeros((self.num_spins, self.num_dt + 1))
        self.Mz = np.zeros((self.num_spins, self.num_dt + 1))

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def random_walk(
        self,
        IV: bool,
        grid: BOLDgrid.Grid3D,
        progressbar: bool=True
    ):

        # populating the inital position of the spins (at t=0)
        self.place_spins(IV, grid, progressbar)

        self.N = grid.N

        # calculate the diffusion distance
        h = np.sqrt(self.ADC * 2 * (self.dt / 1000))

        # randomly positions spins in the voxel
        previous_position = np.empty((self.num_spins, 3))
        previous_position[:, :] = self.positions[:, 0, :]

        # checks if initial spin positions are IV and calculates dBz / phase
        previous_grid_positions = self.position_to_grid(previous_position, grid)
        self.phase[:, 0] = grid.phase[previous_grid_positions]
        self.is_IV[:, 0] = grid.mask[previous_grid_positions]

        # iterating through all time steps
        text = 'Diffusing Spins'
        for j in tqdm(range(1, self.num_dt), desc=text, disable=not progressbar):

            # calculate a new position for all spins (random walk)
            new_position = previous_position + h * self.rng.normal(size=(self.num_spins, 3))
            new_position += grid.size * \
                (1 * (new_position < -0.5 * grid.size) -
                    1 * (new_position > 0.5 * grid.size))

            # checks if new spin positions are IV and calculates dBz /
            # phase
            self.is_IV[:, j] = grid.mask[self.position_to_grid(new_position, grid)]

            # finds all spins that pass through a vessel wall (diffusing)
            diffusing_spin_indices = ((self.is_IV[:, j-1] != self.is_IV[:, j])).nonzero()[0]
            
            # iterates through all diffusing spins
            for i in diffusing_spin_indices:
                # finds if the spin was IV before and after the diffusion
                # through the vessel wall
                prev_is_IV = self.is_IV[i, j-1] > 0

                # finds the vessel object whose wall the spin is crossing
                if prev_is_IV:
                    vsl_number = self.is_IV[i, j-1]
                else:
                    vsl_number = self.is_IV[i, j]

                # create a bool according to the probability (if True the
                # spin is allowed to diffuse through the vessel wall)
                diffuses = self.rng.random() < grid.permeation_probability_list[vsl_number-1]

                # if the spin is not allowed to diffuse through the vessel
                # wall
                if not diffuses:
                    # calculate new position until it does not diffuse
                    # across a vessel wall
                    is_diffused = True
                    while is_diffused:
                        new_position[i, :] = previous_position[i, :] + h * self.rng.normal(size=(3))
                        new_position[i,:] += grid.size * \
                            (1 * (new_position[i,:] < -0.5 * grid.size) - \
                                1 * (new_position[i,:] > 0.5 * grid.size))

                        is_diffused = self.check_diffusion(new_position[i, :], self.is_IV[i, j-1], grid)

            new_grid_position = self.position_to_grid(new_position, grid)
            self.is_IV[:, j] = grid.mask[new_grid_position]
            self.phase[:, j] = grid.phase[new_grid_position]

            # new position becomes the previous position
            previous_position = new_position 

    def check_diffusion(
        self,
        new_pos: np.ndarray,
        prev_is_IV_number: int,
        grid: BOLDgrid.Grid3D
    ):
        # check if the spin has diffused in any vessels
        position = np.expand_dims(new_pos, axis=0)
        is_diffused = prev_is_IV_number != grid.mask[self.position_to_grid(position, grid)]
        return is_diffused 

    def place_spins(
        self,
        IV: bool,
        grid: BOLDgrid.Grid3D,
        progressbar: bool=True
    ):
        if IV:
            # creating extra- and intra-vascular spins
            self.positions[:, 0, :] = grid.size * \
                (self.rng.random((self.num_spins, 3)) - 0.5)

        elif not IV:
            position = grid.size * (self.rng.random((self.num_spins, 3)) - 0.5)
            grid_pos = self.position_to_grid(position, grid)

            is_IV = grid.mask[grid_pos] > 0
            
            IV_spins = is_IV.nonzero()[0]

            text = 'Creating Spins'
            for i in tqdm(IV_spins, desc=text, disable=not progressbar):
                is_IV_fix = True
                while is_IV_fix:
                    # generate position
                    initial_position = grid.size * (self.rng.random((1, 3)) - 0.5)
                    # check if position is in any vessel
                    is_IV_fix = np.all(grid.mask[ self.position_to_grid(initial_position, grid) ] > 0)

                position[i, :] = initial_position

            self.positions[:, 0, :] = position

    def position_to_grid(
        self,
        positions: np.ndarray,
        grid: BOLDgrid.Grid3D
    ):
        subvoxel_per_mm = self.N / grid.size
        grid_positions = ((positions + 0.5 * grid.size ) * subvoxel_per_mm).astype(int)
        X = grid_positions[:,0]
        Y = grid_positions[:,1]
        Z = grid_positions[:,2]

        return X, Y, Z

    def sample_region(self, grid: BOLDgrid.Grid3D):
        # calculates the sample region edge position (center of region at
        # 0,0,0)
        sample_size = (grid.size - 2 * self.edge_width * grid.size) / 2

        # create the boolean array
        self.sample = np.concatenate(
            (np.ones((self.num_spins, 1)), 1 * np.all(np.abs(self.positions) <= sample_size, 2)), 1)

class SpinsContinuous2D:

    def __init__(
        self,
        ADC: float,
        B0: float,
        num_spins: int,
        num_dt: int,
        dt: float,
        edge_width: float,
        seed: int=None
    ):

        self.ADC = ADC
        self.B0 = B0
        self.num_spins = num_spins
        self.num_dt = num_dt
        self.dt = dt
        self.edge_width = edge_width

        self.positions = np.zeros((self.num_spins, self.num_dt, 2))
        self.is_IV_vessel = None
        self.sample = None

        self.phase = np.zeros((self.num_spins, self.num_dt))
        self.is_IV = np.zeros((self.num_spins, self.num_dt), dtype=int)
        self.Mx = np.zeros((self.num_spins, self.num_dt + 1))
        self.My = np.zeros((self.num_spins, self.num_dt + 1))
        self.Mz = np.zeros((self.num_spins, self.num_dt + 1))

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def random_walk(
        self,
        voxel: BOLDvoxel.Voxel2D,
        IV: bool=True,
        record_is_IV_vessel: bool=False, 
        progressbar: bool=True
    ) -> None:
        if record_is_IV_vessel:
            self.is_IV_vessel = np.empty((self.num_spins, self.num_dt), dtype=object)

        # setting the conversion factor for frequency to phase
        phase_conversion_factor = 2 * np.pi * GYROMAGNETIC_RATIO * self.dt * 0.001

        # sets the function to check for vessel wall diffusion
        if IV:
            check_diffusion = self.check_diffusion_EVIV
        elif not IV:
            check_diffusion = self.check_diffusion_EV

        # populating the inital position of the spins (at t=0)
        self.place_spins(voxel, IV, progressbar)

        # calculate the diffusion distance
        h = np.sqrt(self.ADC * 2 * (self.dt / 1000))

        # randomly positions spins in the voxel
        previous_position = np.empty((self.num_spins, 2))
        previous_position[:, :] = self.positions[:, 0, :]

        # initialize the array which will contain information about what spins
        # are in which vessels (IV spins)
        previous_is_IV_vsl = np.full(self.num_spins, None, dtype=object)

        dBz = np.zeros(self.num_spins)
        # checks if initial spin positions are IV and calculates dBz / phase
        for vessel in voxel.vessels:
            is_IV, dBz_EV, dBz_IV = vessel.dBz_mask_from_positions(previous_position, self.B0)
            is_IV = np.logical_and(is_IV, (previous_is_IV_vsl == None))

            previous_is_IV_vsl[is_IV] = vessel
            dBz += np.logical_not(is_IV)*dBz_EV + is_IV*dBz_IV

        self.is_IV[:, 0] = previous_is_IV_vsl != None
        self.phase[:, 0] = phase_conversion_factor * dBz

        if record_is_IV_vessel:
                self.is_IV_vessel[:, 0] = previous_is_IV_vsl

        # iterating through all time steps
        text = 'Diffusing Spins'
        for j in tqdm(range(1, self.num_dt), desc=text, disable=not progressbar):

            # calculate a new position for all spins (random walk)
            new_position = previous_position + h * self.rng.normal(size=(self.num_spins, 2))
            new_position += voxel.size * \
                (1 * (new_position < -0.5 * voxel.size) -
                    1 * (new_position > 0.5 * voxel.size))

            # checks if new spin positions are IV and calculates dBz /
            # phase
            new_is_IV_vsl = np.full(self.num_spins, None, dtype=object)
            dBz = np.zeros(self.num_spins)

            for vessel in voxel.vessels:
                is_IV, dBz_EV, dBz_IV = vessel.dBz_mask_from_positions(previous_position, self.B0)
                is_IV = np.logical_and(is_IV, (new_is_IV_vsl == None))

                new_is_IV_vsl[is_IV] = vessel
                dBz += np.logical_not(is_IV) * dBz_EV + is_IV * dBz_IV


            self.phase[:, j] = phase_conversion_factor * dBz

            # finds all spins that pass through a vessel wall (diffusing)
            permeating_spins_indices = ((previous_is_IV_vsl != new_is_IV_vsl)).nonzero()[0]

            # initialize some lists for patching incorrect diffusion
            # positions
            fix_spins_indices = []
            fix_spins_positions = []
            fix_spins_is_IV_vsl = []

            # iterates through all diffusing spins
            for i in permeating_spins_indices:
                # finds if the spin was IV before and after the diffusion
                # through the vessel wall
                previous_is_IV = previous_is_IV_vsl[i] is not None

                # finds the vessel object whose wall the spin is crossing
                if previous_is_IV:
                    vessel = previous_is_IV_vsl[i]
                else:
                    vessel = new_is_IV_vsl[i]

                # create a bool according to the probability (if True the
                # spin is allowed to diffuse through the vessel wall)
                diffuses = self.rng.random() < vessel.permeation_probability

                # if the spin is not allowed to diffuse through the vessel
                # wall
                if not diffuses:
                    # calculate new position until it does not diffuse
                    # across a vessel wall
                    is_diffused = True
                    while is_diffused:
                        new_position[i, :] = previous_position[i, :] + h * self.rng.normal(size=(2))
                        new_position[i,:] += voxel.size * \
                            (1 * (new_position[i,:] < -0.5 * voxel.size) - \
                                1 * (new_position[i,:] > 0.5 * voxel.size))

                        is_diffused = check_diffusion(
                            voxel,
                            vessel,
                            new_position[i, :],
                            previous_is_IV
                        )

                    new_is_IV_vsl[i] = previous_is_IV_vsl[i]

                    fix_spins_indices.append(i)
                    fix_spins_positions.append(new_position[i, :])
                    fix_spins_is_IV_vsl.append(previous_is_IV_vsl[i])

            #convert list to array for convenience
            fix_spins_indices = np.array(fix_spins_indices)
            fix_spins_positions = np.array(fix_spins_positions)
            fix_spins_is_IV_vsl = np.array(fix_spins_is_IV_vsl)

            #recalculate the phase of the incorrect spins
            if fix_spins_indices.size > 0:
                dBz = self.dBz_fix(fix_spins_positions, fix_spins_is_IV_vsl, voxel)
                self.phase[fix_spins_indices,j] = \
                    phase_conversion_factor * dBz

            # assign the new positions
            self.is_IV[:, j] = new_is_IV_vsl != None
            self.positions[:, j, :] = new_position

            if record_is_IV_vessel:
                self.is_IV_vessel[:, j] = new_is_IV_vsl

            # new position becomes the previous position
            previous_position = new_position
            previous_is_IV_vsl = new_is_IV_vsl

    def dBz_fix(
        self,
        positions: np.ndarray,
        is_IV_vsl: np.ndarray,
        voxel: BOLDvoxel.Voxel2D
    ) -> np.ndarray:
        dBz = np.zeros(is_IV_vsl.shape)

        # iterating through all vessels
        for vsl in voxel.vessels:
            # calculate the EV contribution of the vessel
            dBz += (is_IV_vsl != vsl) * vsl.dBz_EV(positions, self.B0) + \
                (is_IV_vsl == vsl) * vsl.dBz_IV(self.B0)

        return dBz

    def check_diffusion_EVIV(
        self,
        voxel: BOLDvoxel.Voxel2D,
        vessel: BOLDvessel.Vessel2D,
        position: np.ndarray,
        previous_is_IV: bool
    ) -> bool:
        # check if the spin has diffused in the specific vessel only
        position = np.expand_dims(position, axis=0)
        is_permeated = vessel.is_IV(position) != previous_is_IV
        return(is_permeated)
        

    def check_diffusion_EV(
        self,
        voxel: BOLDvoxel.Voxel2D,
        vessel: BOLDvessel.Vessel2D,
        position: np.ndarray,
        previous_is_IV: bool
    ) -> bool:
        # check if the spin has diffused in any vessels
        position = np.expand_dims(position, axis=0)
        for vessel in voxel.vessels:
            is_permeated = vessel.is_IV(position) != previous_is_IV
            if is_permeated:  # if the spin has diffused across a vessel wall, try again
                break
        return(is_permeated)

    def place_spins(
        self,
        voxel: BOLDvoxel.Voxel2D,
        IV: bool,
        progressbar: bool=True
    ) -> None:

        if IV:
            # creating extra- and intra-vascular spins
            self.positions[:, 0, :] = voxel.size * \
                (self.rng.random((self.num_spins, 2)) - 0.5)

        elif not IV:
            position = voxel.size * (self.rng.random((self.num_spins, 2)) - 0.5)

            is_IV = np.zeros(self.num_spins)
            for vessel in voxel.vessels:
                is_IV += vessel.is_IV(position)

            IV_spins = is_IV.nonzero()[0]

            text = 'Creating Spins'
            for i in tqdm(IV_spins, desc=text, disable=not progressbar):
                is_IV_fix = True
                while is_IV_fix:
                    # generate position
                    initial_position = voxel.size * (self.rng.random((1, 2)) - 0.5)
                    # check if position is in any vessel
                    for vessel in voxel.vessels:
                        is_IV_fix = vessel.is_IV(initial_position)
                        if is_IV_fix:
                            break

                position[i, :] = initial_position

            self.positions[:, 0, :] = position

    def sample_region(
        self,
        voxel: BOLDvoxel.Voxel2D
    ) -> None:

        # calculates the sample region edge position (center of region at
        # 0,0,0)
        sample_size = (voxel.size - 2 * self.edge_width * voxel.size) / 2

        # create the boolean array
        self.sample = np.concatenate(
            (np.ones((self.num_spins, 1)), 1 * np.all(np.abs(self.positions) <= sample_size, 2)), 1)

class SpinsDiscrete2D:

    def __init__(
        self,
        ADC: float,
        B0: float,
        num_spins: int,
        num_dt: int,
        dt: float,
        edge_width: float,
        seed: int=None
    ):

        self.ADC = ADC
        self.B0 = B0
        self.num_spins = num_spins
        self.num_dt = num_dt
        self.dt = dt
        self.N = None
        self.edge_width = edge_width

        self.positions = np.zeros((self.num_spins, self.num_dt, 2))
        self.sample = None

        self.phase = np.zeros((self.num_spins, self.num_dt))
        self.is_IV = np.zeros((self.num_spins, self.num_dt), dtype=int)
        self.Mx = np.zeros((self.num_spins, self.num_dt + 1))
        self.My = np.zeros((self.num_spins, self.num_dt + 1))
        self.Mz = np.zeros((self.num_spins, self.num_dt + 1))

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def random_walk(
        self,
        IV: bool,
        grid: BOLDgrid.Grid2D,
        progressbar: bool=True
    ):

        # populating the inital position of the spins (at t=0)
        self.place_spins(IV, grid, progressbar)

        self.N = grid.N

        # calculate the diffusion distance
        h = np.sqrt(self.ADC * 2 * (self.dt / 1000))

        # randomly positions spins in the voxel
        previous_position = np.empty((self.num_spins, 2))
        previous_position[:, :] = self.positions[:, 0, :]

        # checks if initial spin positions are IV and calculates dBz / phase
        previous_grid_position = self.position_to_grid(previous_position, grid)
        self.phase[:, 0] = grid.phase[previous_grid_position]
        self.is_IV[:, 0] = grid.mask[previous_grid_position]

        # iterating through all time steps
        text = 'Diffusing Spins'
        for j in tqdm(range(1, self.num_dt), desc=text, disable=not progressbar):

            # calculate a new position for all spins (random walk)
            new_position = previous_position + h * self.rng.normal(size=(self.num_spins, 2))
            new_position += grid.size * \
                (1 * (new_position < -0.5 * grid.size) -
                    1 * (new_position > 0.5 * grid.size))

            # checks if new spin positions are IV and calculates dBz /
            # phase
            self.is_IV[:, j] = grid.mask[self.position_to_grid(new_position, grid)]

            # finds all spins that pass through a vessel wall (diffusing)
            diffusing_spin_indices = ((self.is_IV[:, j-1] != self.is_IV[:, j])).nonzero()[0]
            
            # iterates through all diffusing spins
            for i in diffusing_spin_indices:
                # finds if the spin was IV before and after the diffusion
                # through the vessel wall
                previous_is_IV = self.is_IV[i, j-1] > 0

                # finds the vessel object whose wall the spin is crossing
                if previous_is_IV:
                    vsl_number = self.is_IV[i, j-1]
                else:
                    vsl_number = self.is_IV[i, j]

                # create a bool according to the probability (if True the
                # spin is allowed to diffuse through the vessel wall)
                diffuses = self.rng.random() < grid.permeation_probability_list[vsl_number-1]

                # if the spin is not allowed to diffuse through the vessel
                # wall
                if not diffuses:
                    # calculate new position until it does not diffuse
                    # across a vessel wall
                    is_diffused = True
                    while is_diffused:
                        new_position[i, :] = previous_position[i, :] + h * self.rng.normal(size=(2))
                        new_position[i,:] += grid.size * \
                            (1 * (new_position[i,:] < -0.5 * grid.size) - \
                                1 * (new_position[i,:] > 0.5 * grid.size))

                        is_diffused = self.check_diffusion(new_position[i, :], self.is_IV[i, j-1], grid)

            new_grid_position = self.position_to_grid(new_position, grid)
            self.is_IV[:, j] = grid.mask[new_grid_position]
            self.phase[:, j] = grid.phase[new_grid_position]

            # assign the new positions
            self.positions[:, j, :] = new_position

            # new position becomes the previous position
            previous_position = new_position 

    def check_diffusion(
        self,
        new_pos: np.ndarray,
        prev_is_IV_number: int,
        grid: BOLDgrid.Grid2D
    ):
        # check if the spin has diffused in any vessels
        position = np.expand_dims(new_pos, axis=0)
        is_diffused = prev_is_IV_number != grid.mask[self.position_to_grid(position, grid)]
        return is_diffused 

    def place_spins(
        self,
        IV: bool,
        grid: BOLDgrid.Grid2D,
        progressbar: bool=True
    ):
        if IV:
            # creating extra- and intra-vascular spins
            self.positions[:, 0, :] = grid.size * \
                (self.rng.random((self.num_spins, 2)) - 0.5)

        elif not IV:
            position = grid.size * (self.rng.random((self.num_spins, 2)) - 0.5)
            grid_position = self.position_to_grid(position, grid)

            is_IV = grid.mask[grid_position] > 0
            
            IV_spins = is_IV.nonzero()[0]

            text = 'Creating Spins'
            for i in tqdm(IV_spins, desc=text, disable=not progressbar):
                is_IV_fix = True
                while is_IV_fix:
                    # generate position
                    initial_position = grid.size * (self.rng.random((1, 2)) - 0.5)
                    # check if position is in any vessel
                    is_IV_fix = np.all(grid.mask[ self.position_to_grid(initial_position, grid) ] > 0)

                position[i, :] = initial_position

            self.positions[:, 0, :] = position

    def position_to_grid(
        self,
        positions: np.ndarray,
        grid: BOLDgrid.Grid2D
    ):
        subvoxel_per_mm = self.N / grid.size
        grid_positions = ((positions + 0.5 * grid.size ) * subvoxel_per_mm).astype(int)
        X = grid_positions[:,0]
        Y = grid_positions[:,1]

        return X, Y

    def sample_region(self, grid: BOLDgrid.Grid2D):
        # calculates the sample region edge position (center of region at
        # 0,0,0)
        sample_size = (grid.size - 2 * self.edge_width * grid.size) / 2

        # create the boolean array
        self.sample = np.concatenate(
            (np.ones((self.num_spins, 1)), 1 * np.all(np.abs(self.positions) <= sample_size, 2)), 1)
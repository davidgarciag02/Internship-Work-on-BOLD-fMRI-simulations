import warnings
from . import BOLDparameters, BOLDspinsDeprecated, BOLDvessel, BOLDvoxel, BOLDgrid, BOLDsequence, BOLDdeterministic
from .BOLDconstants import *
import numpy as np
import datetime
from typing import Optional, List

class SimulationContinuous3D:

    def __init__(
        self,
        parameters: BOLDparameters.ParametersContinuous3D,
        IV: bool=True,
        vessels: Optional[List[BOLDvessel.Vessel3D]] = None,
        voxelseed: int=None,
        spinseed: int=None,
        progressbar: bool=True
    ):
        self.IV = IV
        self.voxelseed = voxelseed
        self.spinseed = spinseed

        self.parameters = parameters
        self.progressbar = progressbar

        self.validate_parameters(parameters)

        if vessels is None:
            self.voxel = BOLDvoxel.Voxel3D.from_random(
                size=parameters.size,
                CBV=self.parameters.CBV,
                labels=self.parameters.labels,
                id_weights=self.parameters.id_weights,
                id_diameters=self.parameters.id_diameters,
                id_dchis=self.parameters.id_dchis,
                id_permeation_probabilities=self.parameters.id_permeation_probabilities,
                vessel_type=self.parameters.vessel_type,
                allow_vessel_intersection=self.parameters.allow_vessel_intersection,
                seed=voxelseed,
                progressbar=self.progressbar
            )
        else:
            self.voxel = BOLDvoxel.Voxel3D(
                size=parameters.size,
                vessels=vessels,
                seed=None
            )
        
        self.spins = BOLDspinsDeprecated.Spins(
            ADC=self.parameters.ADC,
            B0=self.parameters.B0,
            num_spins=self.parameters.num_spins,
            num_dt=self.parameters.num_dt,
            dt=self.parameters.dt,
            seed=self.spinseed
        )

        self.sequence = BOLDsequence.Sequence(
            pulse_time_indices=self.parameters.pulse_time_indices,
            pulse_angles=self.parameters.pulse_angles,
            pulse_axes=self.parameters.pulse_axes,
            dB0=self.parameters.dB0,
            dB1fact=self.parameters.dB1fact,
            T2EV=self.parameters.T2EV,
            T2IV=self.parameters.T2IV,
            T1EV=self.parameters.T1EV,
            T1IV=self.parameters.T1IV           
        )

    def signal_simulation(
        self,
        cplx: bool=False,        
    ):

        self.spins.random_walk(
            voxel=self.voxel, 
            IV=self.IV,
            progressbar=self.progressbar
        )

        signal = self.sequence.signal(
            phase=self.spins.phase,
            is_IV=self.spins.is_IV,
            num_samples=self.spins.num_spins,
            num_dt=self.spins.num_dt,
            dt=self.spins.dt,
            sample_mask=self.spins.sample,
            cplx=cplx
        )

        time_range = np.arange(0, self.parameters.num_dt * self.parameters.dt +
                         self.parameters.dt, self.parameters.dt)

        return signal, time_range

    def validate_parameters(
        self,
        parameters: BOLDparameters.ParametersContinuous3D
    ):
        #TODO add validation
        pass

class SimulationDiscrete3D:

    def __init__(
        self,
        parameters: BOLDparameters.ParametersDiscrete3D,
        IV: bool=True,
        voxelseed: int=None,
        spinseed: int=None
    ):
        self.IV = IV
        self.voxelseed = voxelseed
        self.spinseed = spinseed
        
        self.parameters = parameters
        self.validate_parameters(parameters)

        self.grid = None
        
        self.spins = BOLDspinsDeprecated.SpinsDiscrete3D(
            ADC=self.parameters.ADC,
            B0=self.parameters.B0,
            num_spins=self.parameters.num_spins,
            num_dt=self.parameters.num_dt,
            dt=self.parameters.dt,
            seed=spinseed
        )
        self.sequence = BOLDsequence.Sequence(
            pulse_time_indices=self.parameters.pulse_time_indices,
            pulse_angles=self.parameters.pulse_angles,
            pulse_axes=self.parameters.pulse_axes,
            dB0=self.parameters.dB0,
            dB1fact=self.parameters.dB1fact,
            T2EV=self.parameters.T2EV,
            T2IV=self.parameters.T2IV,
            T1EV=self.parameters.T1EV,
            T1IV=self.parameters.T1IV           
        )

        self.voxel = None


    def signal_simulation(
        self,
        offset_method: str='AnalyticalFromVoxel',
        cplx: bool=False,
        progressbar: bool=True,
        file: Optional[str]=None,
        vessels: Optional[List[BOLDvessel.Vessel3D]] = None,
        extend: bool=False
    ):
    
        if offset_method in ('AnalyticalFromVoxel', 'FFTFromVoxel'):
            if vessels is None:
                self.voxel = BOLDvoxel.Voxel3D.from_random(
                    size=self.parameters.size,
                    CBV=self.parameters.CBV,
                    labels=self.parameters.labels,
                    id_weights=self.parameters.id_weights,
                    id_diameters=self.parameters.id_diameters,
                    id_dchis=self.parameters.id_dchis,
                    id_permeation_probabilities=self.parameters.id_permeation_probabilities,
                    vessel_type=self.parameters.vessel_type,
                    allow_vessel_intersection=self.parameters.allow_vessel_intersection,
                    seed=self.voxelseed,
                    progressbar=progressbar
                )
            else:
                self.voxel = BOLDvoxel.Voxel3D(
                    size=self.parameters.size,
                    vessels=vessels
                )

        if offset_method == 'AnalyticalFromVoxel':
            self.grid = BOLDgrid.Grid3D.from_voxel_analytical(
                B0=self.parameters.B0,
                dt=self.parameters.dt,
                N=self.parameters.N,
                voxel=self.voxel,
                progressbar=progressbar
            )
        elif offset_method == 'FFTFromVoxel':
            self.grid = BOLDgrid.Grid3D.from_voxel_FFT(
                B0=self.parameters.B0,
                dt=self.parameters.dt,
                N=self.parameters.N,
                voxel=self.voxel,
                extend=extend,
                padding=self.parameters.padding,
                progressbar=progressbar
            )
        elif offset_method == 'FFTFromFile':

            dchi_list = [self.parameters.id_dchis[str(i)] for i in len(self.parameters.id_dchis)]
            permeation_probability_list = [self.parameters.id_permeation_probabilities[str(i)] for i in len(self.parameters.id_permeation_probabilities)]

            self.grid = BOLDgrid.Grid3D.from_file_FFT(
                filepath=file, 
                dchi_list=dchi_list,
                permeation_probability_list=permeation_probability_list,
                size=self.parameters.size,
                dt=self.parameters.dt,
                B0=self.parameters.B0,
                padding=self.parameters.padding
            )

        else:
            raise Exception(f'Offset method \'{offset_method}\' is not valid!')

        self.spins.random_walk(self.IV, self.grid, progressbar)
        self.spins.sample_region(self.grid)
        
        signal = self.sequence.signal(
            phase=self.spins.phase,
            is_IV=self.spins.is_IV,
            num_samples=self.spins.num_spins,
            num_dt=self.spins.num_dt,
            dt=self.spins.dt,
            sample_mask=self.spins.sample,
            cplx=cplx
        )

        time_range = np.arange(0, self.parameters.num_dt * self.parameters.dt +
                         self.parameters.dt, self.parameters.dt)

        return signal, time_range

    def validate_parameters(
        self,
        parameters: BOLDparameters.ParametersDiscrete3D
    ):
        #TODO add validation
        pass

class SimulationContinuous2D:
    
    def __init__(
        self,
        parameters: BOLDparameters.ParametersContinuous2D,
        IV: bool=True,
        voxelseed: int=None,
        spinseed: int=None
    ):

        self.parameters = parameters

        self.validate_parameters(parameters)
        self.IV=IV #not used for deterministic
        
        self.voxelseed = voxelseed
        self.spinseed = spinseed #not used for deterministic
        self.type = None
        
        self.params = parameters
        
        self.voxel = BOLDvoxel.Voxel2D(
            size=self.parameters.size,
            seed=voxelseed
        )
        
        self.spins=BOLDspinsDeprecated.SpinsContinuous2D(
            ADC=self.parameters.ADC,
            B0=self.parameters.B0,
            num_spins=self.parameters.num_spins,
            num_dt=self.parameters.num_dt,
            dt=self.parameters.dt,
            seed=spinseed            
        )

        self.sequence = BOLDsequence.Sequence(
            pulse_time_indices=self.parameters.pulse_time_indices,
            pulse_angles=self.parameters.pulse_angles,
            pulse_axes=self.parameters.pulse_axes,
            dB0=self.parameters.dB0,
            dB1fact=self.parameters.dB1fact,
            T2EV=self.parameters.T2EV,
            T2IV=self.parameters.T2IV,
            T1EV=self.parameters.T1EV,
            T1IV=self.parameters.T1IV           
        )
    
    
    def signal_simulation(
        self, 
        simulation_type: str='MonteCarlo',
        progressbar: bool=True, 
        cplx: bool=False,
        vessels: Optional[List[BOLDvessel.Vessel2D]]=None
    ):

        start = datetime.datetime.now()
        
        if vessels is None:
            self.voxel.random_populate(
                CBV=self.parameters.CBV,
                labels=self.parameters.labels,
                id_weights=self.parameters.id_weights,
                id_diameters=self.parameters.id_diameters,
                id_dchis=self.parameters.id_dchis,
                id_permeation_probabilities=self.parameters.id_permeation_probabilities,
                vessel_type=self.parameters.vessel_type,
                allow_vessel_intersection=self.parameters.allow_vessel_intersection,
                progressbar=progressbar
            )
        else:
            self.voxel.vessels = vessels

        if simulation_type == 'MonteCarlo':
            self.spins.random_walk(self.voxel, self.IV, progressbar)

            self.spins.sample_region(self.voxel)
                
            signal = self.sequence.signal(
                phase=self.spins.phase,
                is_IV=self.spins.is_IV,
                num_samples=self.parameters.num_spins,
                num_dt=self.parameters.num_dt,
                dt=self.parameters.dt,
                sample_mask=self.spins.sample,
                cplx=cplx
            )
        
        elif simulation_type == 'MonteCarlo3B0':

            #make vessels perpendicular to B0
            for vsl in self.voxel.vessels:
                vsl.theta = np.pi/2
                vsl.phi = 0

            self.spins.random_walk(self.voxel, self.IV, True, progressbar)
            
            self.spins.phase *= 2/3

            #make vessels parallel
            for vsl in self.voxel.vessels:
                vsl.theta = 0
                vsl.phi = 0

            for vsl in self.voxel.vessels:                
                self.spins.phase[self.spins.is_IV_vessel == vsl] += 1/3 * vsl.dBz_IV(self.spins.B0) * (2*np.pi*GYROMAGNETIC_RATIO*self.parameters.dt*0.001)

            signal = self.sequence.signal(
                phase=self.spins.phase,
                is_IV=self.spins.is_IV,
                num_samples=self.parameters.num_spins,
                num_dt=self.parameters.num_dt,
                dt=self.parameters.dt,
                sample_mask=self.spins.sample,
                cplx=cplx
            )
        else:
            raise Exception(f'Simulation Type \'{simulation_type}\' is not valid!')

            
        end = datetime.datetime.now()
        
        trng = np.arange(0, self.params.num_dt*self.params.dt+self.params.dt, self.params.dt)
        simTime = end - start
           
        return signal, trng, simTime

    def validate_parameters(
        self,
        parameters: BOLDparameters.Parameters2D
    ):
        #TODO add validation
        pass

class SimulationDiscrete2D:
    
    def __init__(
        self,
        parameters: BOLDparameters.ParametersDiscrete2D,
        IV: bool=True,
        voxelseed: int=None,
        spinseed: int=None
    ):
        self.IV = IV
        self.voxelseed = voxelseed
        self.spinseed = spinseed
        
        self.parameters = parameters
        self.validate_parameters(parameters)

        self.grid = BOLDgrid.Grid2D(
            B0=self.parameters.B0,
            dt=self.parameters.dt,
            N=self.parameters.N
        )
        
        #potentially unused, but it is a light structure if not populated
        self.spins = BOLDspinsDeprecated.SpinsDiscrete2D(
            ADC=self.parameters.ADC,
            B0=self.parameters.B0,
            num_spins=self.parameters.num_spins,
            num_dt=self.parameters.num_dt,
            dt=self.parameters.dt,
            seed=spinseed
        )
        
        #potentially unused, but it is a light structure if not populated
        self.deterministic_diffuser = BOLDdeterministic.DeterministicDiffuser2D(
            ADC=self.parameters.ADC,
            B0=self.parameters.B0,
            num_dt=self.parameters.num_dt,
            dt=self.parameters.dt,
        )

        self.sequence = BOLDsequence.Sequence(
            pulse_time_indices=self.parameters.pulse_time_indices,
            pulse_angles=self.parameters.pulse_angles,
            pulse_axes=self.parameters.pulse_axes,
            dB0=self.parameters.dB0,
            dB1fact=self.parameters.dB1fact,
            T2EV=self.parameters.T2EV,
            T2IV=self.parameters.T2IV,
            T1EV=self.parameters.T1EV,
            T1IV=self.parameters.T1IV           
        )

        #potentially unused, but it is a light structure if not populated
        self.voxel = BOLDvoxel.Voxel2D(size=self.parameters.size, seed=voxelseed)


    def signal_simulation(
        self,
        offset_method: str='AnalyticalFromVoxel',
        diffusion_scheme: str='MonteCarlo',
        kernel_type: str='ModifiedBessel',
        permeable=True,
        cplx: bool=False,
        progressbar: bool=True,
        file: Optional[str]=None,
        vessels: Optional[List[BOLDvessel.Vessel3D]] = None,
        extend: bool=False
    ):

        start = datetime.datetime.now()
    
        if offset_method in ('AnalyticalFromVoxel', 'FFTFromVoxel'):
            if vessels is None:
                self.voxel.random_populate(
                    CBV=self.parameters.CBV,
                    labels=self.parameters.labels,
                    id_weights=self.parameters.id_weights,
                    id_diameters=self.parameters.id_diameters,
                    id_dchis=self.parameters.id_dchis,
                    id_permeation_probabilities=self.parameters.id_permeation_probabilities,
                    vessel_type=self.parameters.vessel_type,
                    allow_vessel_intersection=self.parameters.allow_vessel_intersection,
                    progressbar=progressbar
                )
            else:
                self.voxel.vessels = vessels

        if offset_method == 'AnalyticalFromVoxel':
            self.grid.populate(
                mode=offset_method,
                voxel=self.voxel,
                progressbar=progressbar
            )
        elif offset_method == 'FFTFromVoxel':
            self.grid.populate(
                mode=offset_method,
                padding=self.parameters.padding,
                voxel=self.voxel,
                progressbar=progressbar, 
                extend=extend
                )
        elif offset_method == 'FFTFromFile':
            self.grid.populate(
                mode=offset_method,
                size=self.parameters.size,
                padding=self.parameters.padding,
                id_permeation_probabilities=self.parameters.id_permeation_probabilities,
                id_dchis=self.parameters.id_dchis,
                file=file, 
                progressbar=progressbar
            )

        else:
            raise Exception(f'Offset method \'{offset_method}\' is not valid!')

        if diffusion_scheme == 'MonteCarlo':

            self.spins.random_walk(self.IV, self.grid, progressbar)
            self.spins.sample_region(self.grid)
            
            signal = self.sequence.signal(
                phase=self.spins.phase,
                is_IV=self.spins.is_IV,
                num_samples=self.spins.num_spins,
                num_dt=self.spins.num_dt,
                dt=self.spins.dt,
                sample_mask=self.spins.sample,
                cplx=cplx
            )
        
        elif diffusion_scheme == 'Deterministic':
            if self.parameters.id_permeation_probabilities is not None:
                warnings.warn('Permeation probabilities do not work with deterministic simulations! Select either permeable or impermeable.')
            
            signal = self.deterministic_diffuser.signal(
                grid=self.grid,
                sequence=self.sequence,
                kernel_type=kernel_type,
                cplx=cplx,
                permeable=permeable,
                progressbar=progressbar
            )

        end = datetime.datetime.now()

        time_range = np.arange(0, self.parameters.num_dt * self.parameters.dt +
                        self.parameters.dt, self.parameters.dt)
        simulation_time = end - start

        return signal, time_range, simulation_time

    def validate_parameters(
        self,
        parameters: BOLDparameters.ParametersDiscrete3D
    ):
        #TODO add validation
        pass
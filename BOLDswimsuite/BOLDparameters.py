import numpy as np
from dataclasses import dataclass, InitVar
from typing import List, Dict, Optional, Union

@dataclass
class ParametersContinuous3D:

    # VOXEL PARAMETERS
    
    #shape of the vessel
    vessel_type: str  
    #cerebral blood volume (% ratio)
    CBV: float   

    # VESSEL PARAMETERS

    #identifiers of vessels, simply used as keys to dictionaries
    identifiers: List[str]
    #weight of each vessel id for the total CBV
    id_weights: Dict[str, float]
    #vessel diameter distribution of each id (mm)
    id_diameters: Dict[str, List[float]]
    #susceptibility difference (vessel to tissue) of each id in cgs
    id_dchis: Dict[str, float]
    #permeation probability of each id
    id_permeation_probabilities: Dict[str, float]

    # SPIN PARAMETERS

    #diffusion length (mm^2/s)
    ADC: float
    #magnetic field strength (T) 
    B0: float
    #time step (ms)
    dt: float
    #number of time steps
    num_dt: int
    #number of spins
    num_spins: int

    # SEQUENCE PARAMETERS

    #pulse time indices (first pulse always at 0)
    pulse_time_indices: List[int]
    #list of angle of each pulse (radians)
    pulse_angles: List[float]
    #pulse axes in polar coordinates (radians) list of [phi,theta]
    pulse_axes: List[List[float]]

    # OPTIONAL PARAMETERS
    
    #voxel edge length (mm), calculated from <k> value if set to None     
    size: Optional[float] = None    
    #voxel size to largest radius ratio, used to calculate <size> if <size> is set to None
    k: InitVar[Optional[float]] = None
    #sampling region border width (as a fraction of the space - must be 0 <= edgeWidth < 0.5)
    edge_width: float = 0
    #whether to check for vessel intersection during voxel generation
    allow_vessel_intersection: bool = True  
    #intravascular T1 value in ms, ignored if set to None
    T1IV: Optional[float] = None
    #intravascular T2 value in ms, ignored if set to None
    T2IV: Optional[float] = None
    #extravascular T1 value in ms, ignored if set to None
    T1EV: Optional[float] = None
    #extravascular T2 value in ms, ignored if set to None
    T2EV: Optional[float] = None
    #dB1 offset as a fraction of the dephasing angle
    dB1fact: float = 1
    #dB0 offset in T
    dB0: float = 0

    def __post_init__(self, k):
        '''
        Post initialization procedures
        '''

        # If "k" is given and "size" is None, will calculate the size from the
        # Boxerman equation.
        if self.size is None and k is not None:
            
            diam = max([max(self.id_diameters[idn]) for idn in self.identifiers])
            A = np.sqrt(2 * self.ADC * self.dt / 1000)
            
            self.size = 2 * (A + k * diam / 2)

@dataclass
class ParametersDiscrete3D:

    # VOXEL PARAMETERS
    
    #shape of the vessel 
    vessel_type: str   
    #cerebral blood volume (% ratio)
    CBV: float

    # VESSEL PARAMETERS

    #identifiers of vessels, simply used as keys to dictionaries
    identifiers: List[str]
    #weight of each vessel id for the total CBV
    id_weights: Dict[Union[str,int], float]
    #vessel diameter distribution of each id (mm)
    id_diameters: Dict[Union[str,int], List[float]]
    #susceptibility difference (vessel to tissue) of each id in cgs
    id_dchis: Dict[Union[str,int], float]
    #permeation probability of each id
    id_permeation_probabilities: Dict[Union[str,int], float]

    #GRID PARAMETERS

    #grid size, for cubic grid-bound simulations
    N: int

    # SPIN PARAMETERS

    #diffusion length (mm^2/s)
    ADC: float
    #magnetic field strength (T) 
    B0: float
    #time step (ms)
    dt: float
    #number of time steps
    num_dt: int
    #number of spins
    num_spins: int

    # SIGNAL PARAMETERS

    #pulse time indices (first pulse always at 0)
    pulse_time_indices: List[int]
    #list of angle of each pulse (radians)
    pulse_angles: List[float]
    #pulse axes in polar coordinates (radians) list of [phi,theta]
    pulse_axes: List[List[float]]

    # OPTIONAL PARAMETERS

    #padding when performing the FFT convolution (not used otherwise)
    padding: int = 0
    #voxel edge length (mm), calculated from <k> value if set to None     
    size: Optional[float] = None
    #voxel size to largest radius ratio, used to calculate <size> if <size> is set to None
    k: InitVar[float] = None
    #sampling region border width (as a fraction of the space - must be 0 <= edgeWidth < 0.5)
    edge_width: float = 0
    #whether to check for vessel intersection during voxel generation
    allow_vessel_intersection: bool = True  
    #intravascular T1 value in ms, ignored if set to None
    T1IV: Optional[float] = None
    #intravascular T2 value in ms, ignored if set to None
    T2IV: Optional[float] = None
    #extravascular T1 value in ms, ignored if set to None
    T1EV: Optional[float] = None
    #extravascular T2 value in ms, ignored if set to None
    T2EV: Optional[float] = None
    #dB1 offset as a fraction of the dephasing angle
    dB1fact: float = 1
    #dB0 offset in T
    dB0: float = 0

    def __post_init__(self, k: int):
        '''
        Post initialization procedures
        '''

        # If "k" is given and "size" is None, will calculate the size from the
        # Boxerman equation.
        if self.size is None and k is not None:
            
            diam = max([max(self.id_diameters[idn]) for idn in self.identifiers])
            A = np.sqrt(2 * self.ADC * self.dt / 1000)
            
            self.size = 2 * (A + k * diam / 2)

@dataclass
class ParametersContinuous2D:

    # VOXEL PARAMETERS
    
    #shape of the vessel 
    vessel_type: str
    #cerebral blood volume (% ratio)
    CBV: float

    # VESSEL PARAMETERS

    #identifiers of vessels, simply used as keys to dictionaries
    identifiers: List[str]
    #weight of each vessel id for the total CBV
    id_weights: Dict[str, float]
    #vessel diameter distribution of each id (mm)
    id_diameters: Dict[str, List[float]]
    #susceptibility difference (vessel to tissue) of each id in cgs
    id_dchis: Dict[str, float]
    #permeation probability of each id
    id_permeation_probabilities: Dict[str, float]

    # SPIN PARAMETERS

    #diffusion length (mm^2/s)
    ADC: float
    #magnetic field strength (T)
    B0: float
    #time step (ms)
    dt: float
    #number of time steps
    num_dt: int

    #number of spins (MC)
    num_spins: int

    # SIGNAL PARAMETERS

    #pulse time indices (first pulse always at 0)
    pulse_time_indices: List[int]
    #list of angle of each pulse (radians)
    pulse_angles: List[float]
    #pulse axes in polar coordinates (radians) list of [phi,theta]
    pulse_axes: List[List[float]]

    # OPTIONAL PARAMETERS

    #voxel edge length (mm), calculated from <k> value if set to None     
    size: Optional[float] = None     
    #number of vessels
    num_vessels: InitVar[int] = None
    #sampling region border width (as a fraction of the space - must be 0 <= edgeWidth < 0.5)
    edge_width: float = 0
    #whether to check for vessel intersection during voxel generation
    allow_vessel_intersection: bool = True  
    #intravascular T1 value in ms, ignored if set to None
    T1IV: float = None
    #intravascular T2 value in ms, ignored if set to None
    T2IV: float = None
    #extravascular T1 value in ms, ignored if set to None
    T1EV: float = None
    #extravascular T2 value in ms, ignored if set to None
    T2EV: float = None
    
    #dB1 offset as a fraction of the dephasing angle
    dB1fact: float = 1
    #dB0 offset in T
    dB0: float = 0

    def __post_init__(self, num_vessels):
        '''
        Post initialization procedures
        '''

        # If "k" is given and "size" is None, will calculate the size from the
        # Boxerman equation.
        if self.size is None and num_vessels is not None:
            diam = max([max(self.id_diameters[idn]) for idn in self.identifiers])

            vsl_area = np.pi*(diam/2)**2
            self.size = np.sqrt(vsl_area*num_vessels/self.CBV)

@dataclass
class ParametersDiscrete2D:

    # VOXEL PARAMETERS
    
    #shape of the vessel 
    vessel_type: str   
    #cerebral blood volume (% ratio)
    CBV: float

    # VESSEL PARAMETERS

    #identifiers of vessels, simply used as keys to dictionaries
    identifiers: List[str]
    #weight of each vessel id for the total CBV
    id_weights: Dict[Union[str,int], float]
    #vessel diameter distribution of each id (mm)
    id_diameters: Dict[Union[str,int], List[float]]
    #susceptibility difference (vessel to tissue) of each id in cgs
    id_dchis: Dict[Union[str,int], float]
    #permeation probability of each id
    id_permeation_probabilities: Dict[Union[str,int], float]

    #GRID PARAMETERS

    #grid size, for cubic grid-bound simulations
    N: int

    # SPIN PARAMETERS

    #diffusion length (mm^2/s)
    ADC: float
    #magnetic field strength (T) 
    B0: float
    #time step (ms)
    dt: float
    #number of time steps
    num_dt: int
    #number of spins
    num_spins: int

    # SIGNAL PARAMETERS

    #pulse time indices (first pulse always at 0)
    pulse_time_indices: List[int]
    #list of angle of each pulse (radians)
    pulse_angles: List[float]
    #pulse axes in polar coordinates (radians) list of [phi,theta]
    pulse_axes: List[List[float]]

    # OPTIONAL PARAMETERS

    #padding when performing the FFT convolution (not used otherwise)
    padding: int = 0
    #voxel edge length (mm), calculated from <k> value if set to None     
    size: Optional[float] = None
    #number of vessels
    num_vessels: InitVar[int] = None
    #sampling region border width (as a fraction of the space - must be 0 <= edgeWidth < 0.5)
    edge_width: float = 0
    #whether to check for vessel intersection during voxel generation
    allow_vessel_intersection: bool = True  
    #intravascular T1 value in ms, ignored if set to None
    T1IV: Optional[float] = None
    #intravascular T2 value in ms, ignored if set to None
    T2IV: Optional[float] = None
    #extravascular T1 value in ms, ignored if set to None
    T1EV: Optional[float] = None
    #extravascular T2 value in ms, ignored if set to None
    T2EV: Optional[float] = None
    #dB1 offset as a fraction of the dephasing angle
    dB1fact: float = 1
    #dB0 offset in T
    dB0: float = 0

    def __post_init__(self, num_vessels):
        '''
        Post initialization procedures
        '''

        # If "k" is given and "size" is None, will calculate the size from the
        # Boxerman equation.
        if self.size is None and num_vessels is not None:
            diam = max([max(self.id_diameters[idn]) for idn in self.identifiers])

            vsl_area = np.pi*(diam/2)**2
            self.size = np.sqrt(vsl_area*num_vessels/self.CBV)


Parameters3D = Union[ParametersContinuous3D, ParametersDiscrete3D]
Parameters2D = Union[ParametersContinuous2D, ParametersDiscrete2D]
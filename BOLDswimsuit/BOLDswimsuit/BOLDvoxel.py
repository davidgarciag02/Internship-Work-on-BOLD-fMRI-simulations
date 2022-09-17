import numpy as np
from typing import  List, Dict, Optional
from tqdm import tqdm
from . import BOLDvessel

class Voxel3D:
    def __init__(
        self,
        size: float,
        vessels: Optional[List[BOLDvessel.Vessel3D]] = None,
        seed: Optional[int]=None
    ):

        self.generate_random_vessel = {
            'cylinder': self.random_infinite_cylinder_3D,
            'sphere': self.random_sphere_3D,
        }

        # set of specific parameters for a generated voxel
        self.vessels: List[BOLDvessel.Vessel3D] = vessels if vessels is not None else []
        self.size: float = size
        self.rng = np.random.default_rng(seed)

        self.real_CBV: float = None


################################# Infinite Cylinder ######################

    def add_infinite_cylinder_3D(
        self, 
        diameter: float, 
        theta: float, 
        phi: float,
        origin: np.ndarray, 
        dchi: float, 
        permeation_probability: float=0.0, 
        identifier: str='None', 
        volume_percent: float=0.0
    ) -> None:

        #create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(), dtype='uint8')

        self.vessels.append(
            BOLDvessel.InfiniteCylinder3D(
                id_encoded,
                diameter,
                theta,
                phi,
                origin,
                volume_percent,
                dchi,
                permeation_probability
            )
        )

    def random_infinite_cylinder_3D(
        self, 
        identifier: str, 
        diameter: float, 
        dchi: float, 
        permeation_probability: float
    ) -> BOLDvessel.InfiniteCylinder3D:

        # calculate radius of the sphere around the voxel
        voxel_sphere_radius = 0.5 * np.sqrt(3) * self.size

        # generate a random direction for the vessel
        theta = np.arccos(2 * self.rng.random() - 1)
        phi = 2 * np.pi * self.rng.random()

        # generate a random point on the circle slice
        alpha = 2 * np.pi * self.rng.random()
        r = voxel_sphere_radius * np.sqrt(self.rng.random())

        # calculate some trigonometric values in advance, for speed
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        # calculating the origin coordinates
        x = r * (calpha * ctheta * cphi - salpha * sphi)
        y = r * (calpha * ctheta * sphi + salpha * cphi)
        z = -r * calpha * stheta
        origin = np.array([x, y, z])

        # calculate the estimated volume percent of the generated vessel
        height = 2 * np.sqrt(voxel_sphere_radius**2 - r**2)
        total_volume = 4 / 3 * np.pi * voxel_sphere_radius**3
        volume = height * np.pi * (diameter / 2)**2
        volume_percent = volume / total_volume

        # create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(),dtype='uint8')

        # return vessel object with the generated components
        return BOLDvessel.InfiniteCylinder3D(
            id_encoded,
            diameter,
            theta,
            phi,
            origin,
            volume_percent,
            dchi,
            permeation_probability
        )


################################# Sphere #################################

    def add_sphere_3D(
        self,
        diameter: float,
        origin: np.ndarray,
        dchi: float,
        identifier: str='None',
        permeation_probability: float=0.0,
        volume_percent: float=0.0
    ) -> None:
        
        # create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(),dtype='uint8')

        self.vessels.append(
            BOLDvessel.Sphere3D(
                id_encoded,
                diameter,
                origin,
                volume_percent,
                dchi,
                permeation_probability
            )
        )

    def random_sphere_3D(
        self,
        identifier: str,
        diameter: float,
        dchi: float,
        permeation_probability: float
    ) -> BOLDvessel.Sphere3D:

        # generate a random point in the voxel
        origin = (self.rng.random(3) - 0.5) * self.size

        # calculate the estimated volume percent of the generated vessel
        total_volume = self.size**3
        sphere_volume = 4 / 3 * np.pi * (diameter / 2)**3
        volume_percent = sphere_volume / total_volume

        # create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(),dtype='uint8')

        permeation_probability = 0.0 #no IV contribution added, defaults to 0

        # return vessel object with the generated components
        return BOLDvessel.Sphere3D(
            id_encoded,
            diameter,
            origin,
            volume_percent,
            dchi,
            permeation_probability
        )

################################# Populate #################################

    def random_populate(
        self,
        CBV: float,
        identifiers: List[str],
        id_weights: Dict[str, float],
        id_diameters: Dict[str, List[float]],
        id_dchis: Dict[str, float],
        id_permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',
        allow_vessel_intersection: bool = True,
        progressbar: bool=True
    ) -> None:

        total_CBV = 0  # initializing target CBV
        current_CBV = 0  # initializing current CBV
        total_weight = 0 # initializing total CBV weight

        for identifier in identifiers:
            total_weight += id_weights[identifier]

        text = 'Populating Voxel'
        with tqdm(total=100, desc=text, disable=not progressbar) as pbar:

            # iterating through all vessel types
            for identifier in identifiers:
                # CBV occupied by the current vessel type
                type_CBV = CBV * id_weights[identifier] / total_weight
                # target CBV incremented by the vessel type's CBV
                total_CBV += type_CBV

                # generate vessels until the target CBV is reached
                while current_CBV < total_CBV:

                    vessel_intersects = True

                    counter = 0
                    while vessel_intersects:
                        # picks diameter
                        diameters = id_diameters[identifier]
                        diameter = self.rng.choice(diameters)

                        # picks dChi
                        dchi = id_dchis[identifier]

                        # picks permeation probability
                        permeation_probability = id_permeation_probabilities[identifier] if id_permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel = self.generate_random_vessel[vessel_type](
                            identifier, 
                            diameter,
                            dchi, 
                            permeation_probability
                        )                        
                        
                        #check that vessel does not intersect other vessels
                        vessel_intersects = False
                        
                        if not allow_vessel_intersection:
                            for vsl in self.vessels:
                                if vsl.intersects(vessel):
                                    vessel_intersects = True
                                    counter += 1
                                    break

                    # adding the vessel to the list
                    self.vessels.append(vessel)
                    
                    # adding volume contribution from new vessel
                    current_CBV += vessel.volume_percent
                    
                    #manual override of the progress bar
                    progress_percentage = int(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()

            # final CBV stored
            self.real_CBV = current_CBV

class Voxel2D:
    def __init__(
        self,
        size: float,
        seed: Optional[int]=None
    ):
        
        # set of parameters for voxel generation
        self.vessel_type = None

        self.generate_random_vessel = {
            'cylinder': self.random_infinite_cylinder_2D,
        }

        # set of specific parameters for a generated voxel
        self.vessels: List[BOLDvessel.Vessel2D] = []
        self.size = size
        self.real_CBV: float = None
        self.rng = np.random.default_rng(seed)
    

    def add_infinite_cylinder_2D(
        self,
        diameter: float,
        B0_theta: float,
        B0_phi: float,
        origin: np.ndarray,
        dchi: float,
        permeation_probability: float=0.0,
        identifier: str='None',
        volume_percent: float=0.0
    ):

        #create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(),dtype='uint8')

        self.vessels.append(
            BOLDvessel.infiniteCylinder2D(
                id_encoded,
                diameter,
                B0_theta,
                B0_phi,
                origin,
                volume_percent,
                dchi,
                permeation_probability
            )
        )

    def random_infinite_cylinder_2D(
        self,
        identifier: float,
        diameter: float,
        dchi: float,
        permeation_probability: float
    ) -> BOLDvessel.InfiniteCylinder2D:
        origin = (self.rng.random(2)-0.5)*self.size    
        
        total_volume = self.size**2
        vessel_volume = np.pi*(diameter/2)**2
        
        volume_percent = vessel_volume/total_volume
        
        B0_theta=np.arccos(2*self.rng.random()-1)
        B0_phi=2*np.pi*self.rng.random()

        # create vessel identifier in char array format (for Numba)
        id_encoded = np.frombuffer(identifier.encode(), dtype='uint8')
        
        #create a new vessel object using the generated parameters and add it to the vessel list
        return BOLDvessel.InfiniteCylinder2D(
            id_encoded,
            diameter,
            B0_theta,
            B0_phi,
            origin,
            volume_percent,
            dchi,
            permeation_probability,
        )  
        
    def random_populate(
        self,
        CBV: float,
        identifiers: List[str],
        id_weights: Dict[str, float],
        id_diameters: Dict[str, List[float]],
        id_dchis: Dict[str, float],
        id_permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',
        allow_vessel_intersection: bool = True,
        progressbar: bool=True
    ) -> None:

        total_CBV = 0  # initializing target CBV
        current_CBV = 0  # initializing current CBV

        total_weight = 0
        for identifier in identifiers:
            total_weight += id_weights[identifier]

        text = 'Populating Voxel'
        with tqdm(total=100, desc=text, disable=not progressbar) as pbar:
             # iterating through all vessel types
            for identifier in identifiers:
                # CBV occupied by the current vessel type
                type_CBV = CBV * id_weights[identifier] / total_weight
                # target CBV incremented by the vessel type's CBV
                total_CBV += type_CBV

                # generate vessels until the target CBV is reached
                while current_CBV < total_CBV:

                    vessel_intersects = True

                    counter = 0
                    while vessel_intersects:
                        # picks diameter
                        diameters = id_diameters[identifier]
                        diameter = self.rng.choice(diameters)

                        # picks dChi
                        dchi = id_dchis[identifier]

                        # picks permeation probability (impermeable if not set)
                        permeation_probability = id_permeation_probabilities[identifier] if id_permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel = self.generate_random_vessel[vessel_type](
                            identifier, 
                            diameter,
                            dchi, 
                            permeation_probability
                        )                        
                        
                        #check that vessel does not intersect other vessels
                        vessel_intersects = False
                        
                        if not allow_vessel_intersection:
                            for vessel in self.vessels:
                                if vessel.intersects(vessel):
                                    vessel_intersects = True
                                    counter += 1
                                    break

                    # adding the vessel to the list
                    self.vessels.append(vessel)
                    
                    # adding volume contribution from new vessel
                    current_CBV += vessel.volume_percent
                    
                    #manual override of the progress bar
                    progress_percentage = round(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()

            # final CBV stored
            self.real_CBV = current_CBV
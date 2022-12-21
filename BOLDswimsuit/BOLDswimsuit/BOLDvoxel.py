import numpy as np
from typing import  List, Dict, Optional
from tqdm import tqdm
from . import BOLDvessel

class Voxel3D:
    def __init__(
        self,
        size: float,
        vessels: Optional[List[BOLDvessel.Vessel3D]],
        seed: Optional[int]=None
    ):
        self.vessels: List[BOLDvessel.Vessel3D] = vessels if vessels is not None else []
        self.size: float = size
        self.rng = np.random.default_rng(seed)

        self.real_CBV: float = None

    def add_vessel(
        self,
        vessel: BOLDvessel.Vessel3D
    ) -> None:

        self.vessels.append(vessel)

    @classmethod
    def from_random(
        cls,
        size: float,
        CBV: float,
        identifiers: List[str],
        id_weights: Dict[str, float],
        id_diameters: Dict[str, List[float]],
        id_dchis: Dict[str, float],
        id_permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',#type[BOLDvessel.Vessel3D] = BOLDvessel.InfiniteCylinder3D,
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> None:

        voxel = cls(
            size=size,
            vessels=None,
            seed=seed
        )

        str2vessel_class = {
            'cylinder': BOLDvessel.InfiniteCylinder3D,
            'sphere': BOLDvessel.Sphere3D
        }

        vessel_class = str2vessel_class[vessel_type]

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
                        diameter = voxel.rng.choice(diameters)

                        # picks dChi
                        dchi = id_dchis[identifier]

                        # picks permeation probability
                        permeation_probability = id_permeation_probabilities[identifier] if id_permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel = vessel_class.from_random(
                            diameter=diameter,
                            dchi=dchi,
                            voxel_size=voxel.size,
                            permeation_probability=permeation_probability,
                            identifier=identifier,
                            rng=voxel.rng
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
                    current_CBV += vessel.volume_percent
                    
                    #manual override of the progress bar
                    progress_percentage = int(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()

            # final CBV stored
            voxel.real_CBV = current_CBV

            return voxel

class Voxel2D:
    def __init__(
        self,
        size: float,
        vessels: Optional[List[BOLDvessel.Vessel3D]],
        seed: Optional[int]=None
    ):
        self.vessels: List[BOLDvessel.Vessel3D] = vessels if vessels is not None else []
        self.size = size
        self.real_CBV: float = None
        self.rng = np.random.default_rng(seed)
    
    def add_vessel(
        self,
        vessel: BOLDvessel.Vessel2D
    ) -> None:

        self.vessels.append(vessel)

    @classmethod
    def from_random(
        cls,
        size: float,
        CBV: float,
        identifiers: List[str],
        id_weights: Dict[str, float],
        id_diameters: Dict[str, List[float]],
        id_dchis: Dict[str, float],
        id_permeation_probabilities: Optional[Dict[str, float]]=None,
        vessel_type: str='cylinder',#type[BOLDvessel.Vessel3D] = BOLDvessel.InfiniteCylinder3D,
        allow_vessel_intersection: bool = True,
        seed: Optional[int]=None,
        progressbar: bool=True
    ) -> None:

        voxel = cls(
            size=size,
            vessels=None,
            seed=seed
        )

        str2vessel_class = {
            'cylinder': BOLDvessel.InfiniteCylinder3D,
            'sphere': BOLDvessel.Sphere3D
        }

        vessel_class = str2vessel_class[vessel_type]

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
                        diameter = voxel.rng.choice(diameters)

                        # picks dChi
                        dchi = id_dchis[identifier]

                        # picks permeation probability (impermeable if not set)
                        permeation_probability = id_permeation_probabilities[identifier] if id_permeation_probabilities is not None else 0.0

                        #generate vessel
                        vessel = vessel_class.from_random(
                            diameter=diameter,
                            dchi=dchi,
                            voxel_size=voxel.size,
                            permeation_probability=permeation_probability,
                            identifier=identifier,
                            rng=voxel.rng
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
                    current_CBV += vessel.volume_percent
                    
                    #manual override of the progress bar
                    progress_percentage = round(current_CBV / total_CBV * 100) 
                    progress_percentage = 100 if progress_percentage > 100 else progress_percentage
                    pbar.n = progress_percentage
                    pbar.refresh()

            # final CBV stored
            voxel.real_CBV = current_CBV

            return voxel
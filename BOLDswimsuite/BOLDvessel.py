"""
BOLDvessel

Module containing all analytically-defined vessel types.
"""
from __future__ import annotations
import numpy as np
from numba import float64, uint8, types   # import the types
from numba.experimental import jitclass
from typing import Union, Tuple

spec_infinite_cylinder_3D = [
    ('label', types.unicode_type),
    ('diameter', float64),
    ('theta', float64),
    ('phi', float64),
    ('origin', float64[:]),
    ('dchi', float64),
    ('permeation_probability', float64),
    ('normal_vector', float64[:]),
    ('B0_projection_vector', float64[:])
]

@jitclass(spec_infinite_cylinder_3D)
class InfiniteCylinder3DNumba:

    def __init__(
        self,
        label,
        diameter,
        theta, 
        phi, 
        origin,
        dchi, 
        permeation_probability
    ):
        """Analytically-defined infinite cylinder vessel geometry.

        Parameters
        ----------
        label : uint8[:]
            uft-8 encoded string, to identify the vessel.
        diameter : float64
            vessel diameter (mm)
        theta : float64
            zenith angle of the vessel direction (radians)
        phi : float64
            azimuth angle of the vessel direction (radians)
        origin : float64[:]
            cartesian coordinates of the vessel origin (mm)
        dchi : float64
            susceptibility difference between the vessel and the surrounding tissue (cgs units)
        permeation_probability : float64
            probability for a spin to permeate through the vesel wall (fraction of 1)
        """

        # initializing the vessel parameters to the vessel object
        self.label = label
        self.diameter = diameter
        self.theta = theta
        self.phi = phi
        self.origin = origin
        self.dchi = dchi
        self.permeation_probability = permeation_probability
        self.normal_vector = np.array(
            [
                np.sin(self.theta)*np.cos(self.phi),
                np.sin(self.theta)*np.sin(self.phi),
                np.cos(self.theta)
            ]
        )

        # hard coded for [0, 0, 1] B0 direction (for extremely small performance increase)
        B0_direction = np.array([0, 0, 1])
        self.B0_projection_vector = B0_direction-self.normal_vector[2]*self.normal_vector

        # projection for arbitrary B0 is:
        # self.B0_projection_vector = B0_direction - np.dot(B0_direction, self.normal_vector) / (np.linalg.norm(B0_direction * self.normal_vector)) * self.normal_vector

        # causes errors when projection vector is [0,0,0], we can fix it without lost of accuracy by doing the following
        if np.linalg.norm(self.B0_projection_vector) == 0:
            self.B0_projection_vector = np.array([0.0,0.0,1.0])

    def is_IV_dBz(self, positions: np.ndarray, B0: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Given an array of positions and a magnetic field strength for B0, returns whether the positions are intravascular and the dBz magnetic field offset.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            3 element Tuple. The first element is a boolean array, indicating for each position if it is intravascular. The second element is an array with the extravascular dBz magnetic field offset of each position. The thrid element is the intravascular dBz magnetic offset (a single value as it is constant). 
            
            Note that the extravascular dBz is also provided for intravascular positions. In most cases this can be ignored, but is sometimes required during simulations. 
        """      

        # finding the distance between the central axis of the vessel and the point
        radial_vectors = self._radial_vectors_from_positions(positions)

        # finding magnitude of the vectors in radial_distances
        radial_distances = np.sqrt(
            radial_vectors[:, 0]**2 + \
            radial_vectors[:, 1]**2 + \
            radial_vectors[:, 2]**2
        )

        # checking if the positions are IV
        is_IV = radial_distances <= self.diameter/2

        # finding the angle between the projected magnetic field B0 and the point (with respect to the vessel origin)
        cos_phis = np.dot(radial_vectors, np.ascontiguousarray(self.B0_projection_vector)) / \
            (radial_distances*np.linalg.norm(self.B0_projection_vector))
        
        cos2_phis = 2 * cos_phis**2 - 1

        # calculating field offset using the appropriate equation (dchi in cgs, angles in radians, lengths in mm)
        dBz_EV = B0*2*np.pi*self.dchi*(0.5*self.diameter/radial_distances)**2*cos2_phis*(np.sin(self.theta))**2
        
        # some vessel intersection may cause spins to use EV values for dBz even when they are IV
        # this makes all IV values in the dBzEV equal to the value at the vessel boundary where
        # "0.5*self.diameter == radial_distance_magnitudes"
        dBz_EV[is_IV] = B0*2*np.pi*self.dchi*cos2_phis[is_IV]*(np.sin(self.theta))**2

        dBz_IV = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1)

        return is_IV, dBz_EV, dBz_IV

    def is_IV(self, positions: np.ndarray) -> np.ndarray:
        """Given an array of positions, returns whether the positions are intravascular.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Returns
        -------
        np.ndarray
            Boolean array, indicating for each position if it is intravascular.
        """  

        # check if the point(s) are inside the vessel using the cylinder equation
        vectors = self._radial_vectors_from_positions(positions)

        is_IV = vectors[:, 0]**2 + vectors[:, 1]**2 \
            + vectors[:, 2]**2 <= (self.diameter/2)**2
            
        return is_IV

    def grid_is_IV(self, N, subvox_size):
        grid_origin = self.origin / subvox_size + (N/2)

        X = np.reshape(np.arange(N), (N,1,1)) - grid_origin[0]
        Y = np.reshape(np.arange(N), (1,N,1)) - grid_origin[1]
        Z = np.reshape(np.arange(N), (1,1,N)) - grid_origin[2]

        sphi = np.sin(self.phi)
        cphi = np.cos(self.phi)
        stheta = np.sin(self.theta)
        ctheta = np.cos(self.theta)

        grid_radius = (self.diameter/2) / subvox_size

        is_IV = ((X*cphi + Y*sphi)*ctheta - Z*stheta)**2 + (-X*sphi + Y*cphi)**2 <= grid_radius**2

        return is_IV

    def dBz_EV(self, positions: np.ndarray, B0: float) -> np.ndarray:
        """Given an array of positions and a magnetic field strength for B0, returns the extravascular dBz magnetic field offset.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        np.ndarray
            Array with the extravascular dBz magnetic field offset of each position.
            
            Note that the extravascular dBz is also provided for intravascular positions. In most cases this can be ignored, but is sometimes required during simulations. 
        """   

        # finding the distance between the central axis of the vessel and the point
        radial_vectors = self._radial_vectors_from_positions(positions)

        # finding magnitude of the vectors in radial_distances
        radial_distances = np.sqrt(radial_vectors[:, 0]**2 + radial_vectors[:, 1]**2 + radial_vectors[:, 2]**2)

        is_IV = radial_distances <= self.diameter/2

        # finding the angle between the projected magnetic field B0 and the point (with respect to the vessel origin)
        cos_phis = np.dot(radial_vectors, np.ascontiguousarray(self.B0_projection_vector)) / \
            (radial_distances*np.linalg.norm(self.B0_projection_vector))
        cos_2phis = 2 * cos_phis**2 - 1

        # calculating field offset using the appropriate equation (dChi in cgs, angles in radians, lengths in mm)
        dBz = B0*2*np.pi*self.dchi * \
            ((0.5*self.diameter/radial_distances)**2)*cos_2phis*(np.sin(self.theta))**2

        # some vessel intersection may cause spins to use EV values for dBz even when they are IV
        # this makes all IV values in the dBzEV equal to the value at the vessel boundary where
        # "0.5*self.diam == radial_distance_magnitudes"
        dBz[is_IV] = B0*2*np.pi*self.dchi * \
            cos_2phis[is_IV]*(np.sin(self.theta))**2

        return dBz

    def dBz_IV(self, B0: float) -> float:
        """Given a magnetic field strength for B0, returns whether the positions are intravascular and the dBz magnetic field offset.

        Parameters
        ----------
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        float
            The intravascular dBz magnetic offset.
        """
        
        dBz = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1)
        return dBz

    def intersects(self, other: InfiniteCylinder3DNumba) -> bool:
        """Given another 3D infinite cylinder object, returns whether the two vessels intersect.

        Parameters
        ----------
        other : InfiniteCylinder3DNumba
            Another 3D infinite cylinder object.

        Returns
        -------
        bool
            Returns True if the vessels intersect and False otherwise.
        """

        #calculating the shortest distance between the vessel axes
        cross = np.cross(self.normal_vector, other.normal_vector)
        top = np.dot(cross, other.origin - self.origin)
        
        min_distance = np.abs(top) / \
            np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        #checking for intersection
        intersects = min_distance < (self.diameter + other.diameter)/2 
        
        return intersects
    
    def volume_fraction(self, voxel_size: float) -> float:
        """Given the side length of an isometric voxel, returns an estimate of the volume fraction that the vessel occupies in that space.

        Parameters
        ----------
        voxel_size : float
            Side length of the isometric voxel.

        Returns
        -------
        float
            Estimated volume fraction.
        """        
        # calculate radius of the sphere around the voxel
        voxel_sphere_radius = 0.5 * np.sqrt(3) * voxel_size

        radial_vector_to_voxel_center = self._radial_vectors_from_positions(np.array([[0.,0.,0.]]))

        #r = np.linalg.norm(radial_vector_to_voxel_center)
        r = np.linalg.norm(radial_vector_to_voxel_center)
        
        # calculate the estimated volume percent of the generated vessel
        height = 2 * np.sqrt(voxel_sphere_radius**2 - r**2)
        total_volume = 4 / 3 * np.pi * voxel_sphere_radius**3
        volume = height * np.pi * (self.diameter / 2)**2
        volume_fraction = volume / total_volume

        return volume_fraction

    def _radial_vectors_from_positions(self, positions):
        # https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
        
        # vector from origin to positions
        v = positions-self.origin
        # distance between origin and the positions projected onto the vessel axis
        t = np.dot(np.ascontiguousarray(v), np.ascontiguousarray(self.normal_vector))
        # projection of the positions onto the vessel axis
        P = self.origin + np.expand_dims(t, axis=1)*self.normal_vector
        
        return positions-P

class InfiniteCylinder3D:
    """Analytically-defined infinite cylinder vessel geometry. Wrapper for InfiniteCylinder3DNumba, to add functionality that is not compatible with Numba's jitclass.

    Parameters
    ----------
    diameter : float
        Vessel diameter (mm).
    theta : float
        Zenith angle of the vessel direction (radians).
    phi : float
        Azimuth angle of the vessel direction (radians).
    origin : np.ndarray
        Cartesian coordinates of the vessel origin (mm).
    dchi : float
        Susceptibility difference between the vessel and the surrounding tissue (cgs units).
    permeation_probability : float
        Probability for a spin to permeate through the vessel wall (fraction of 1).
    label : str
        String to identify the vessel.
    """
    def __new__(
        cls,
        diameter: float,
        theta: float, 
        phi: float, 
        origin: np.ndarray, 
        dchi: float, 
        permeation_probability: float=0,
        label: str=''
    ):
    
        return InfiniteCylinder3DNumba(
            label=label,
            diameter=diameter,
            theta=theta, 
            phi=phi, 
            origin=origin,
            dchi=dchi, 
            permeation_probability=permeation_probability 
        )

    @staticmethod
    def from_random(
        diameter: float, 
        dchi: float,
        voxel_size: float,
        permeation_probability: float=0,
        label: str='',
        rng: np.random.Generator = np.random.default_rng()
    ) -> InfiniteCylinder3DNumba:
        """Create a vessel with a randomly generated position.

        Parameters
        ----------
        diameter : float
            Vessel diameter (mm).
        dchi : float
            Susceptibility difference between the vessel and the surrounding tissue (cgs units).
        voxel_size : float
            Size of the voxel in which the vessel is positioned, assuming the voxel is centered around zero (mm).
        permeation_probability : float, optional
            Probability for a spin to permeate through the vessel wall (fraction of 1). By default 0.
        label : str, optional
           String to identify the vessel, by default ''.
        rng : np.random.Generator, optional
            Generator object for the random number generation, by default np.random.default_rng().

        Returns
        -------
        InfiniteCylinder3DNumba
        """        

        # calculate radius of the sphere around the voxel
        voxel_sphere_radius = 0.5 * np.sqrt(3) * voxel_size

        # generate a random direction for the vessel
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()

        # generate a random point on the circle slice
        alpha = 2 * np.pi * rng.random()
        r = voxel_sphere_radius * np.sqrt(rng.random())

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

        # return vessel object with the generated components
        return InfiniteCylinder3DNumba(
            label,
            diameter,
            theta,
            phi,
            origin,
            dchi,
            permeation_probability
        )

    def is_IV_dBz(self, positions: np.ndarray, B0: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Given an array of positions and a magnetic field strength for B0, returns whether the positions are intravascular and the dBz magnetic field offset.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            3 element Tuple. The first element is a boolean array, indicating for each position if it is intravascular. The second element is an array with the extravascular dBz magnetic field offset of each position. The thrid element is the intravascular dBz magnetic offset (a single value as it is constant). 
            
            Note that the extravascular dBz is also provided for intravascular positions. In most cases this can be ignored, but is sometimes required during simulations. 
        """
        pass

    def is_IV(self, positions: np.ndarray) -> np.ndarray:
        """Given an array of positions, returns whether the positions are intravascular.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).

        Returns
        -------
        np.ndarray
            Boolean array, indicating for each position if it is intravascular.
        """
        pass

    def dBz_EV(self, positions, B0):
        """Given an array of positions and a magnetic field strength for B0, returns the extravascular dBz magnetic field offset.

        Parameters
        ----------
        positions : np.ndarray
            Array of floats with shape (N, d), where N is the number of positions and d is the number of dimensions (e.g. 2 positions in a 3D space would require an array of shape (2, 3)).
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        np.ndarray
            Array with the extravascular dBz magnetic field offset of each position.
            
            Note that the extravascular dBz is also provided for intravascular positions. In most cases this can be ignored, but is sometimes required during simulations. 
        """
        pass
    
    def dBz_IV(self, B0: float) -> float:
        """Given a magnetic field strength for B0, returns whether the positions are intravascular and the dBz magnetic field offset.

        Parameters
        ----------
        B0 : float
            B0 magnetic field strength (Tesla).

        Returns
        -------
        float
            The intravascular dBz magnetic offset.
        """
        pass

    def intersects(self, other: InfiniteCylinder3DNumba) -> bool:
        """Given another 3D infinite cylinder object, returns whether the two vessels intersect.

        Parameters
        ----------
        other : InfiniteCylinder3DNumba
            Another 3D infinite cylinder object.

        Returns
        -------
        bool
            Returns True if the vessels intersect and False otherwise.
        """
        pass

    def volume_fraction(self, voxel_size: float) -> float:
        """Given the side length of an isometric voxel, returns an estimate of the volume fraction that the vessel occupies in that space.

        Parameters
        ----------
        voxel_size : float
            Side length of the isometric voxel.

        Returns
        -------
        float
            Estimated volume fraction.
        """
        pass       


spec_infinite_cylinder_2D = [
    ('label', types.unicode_type),
    ('diameter', float64),
    ('theta', float64),
    ('phi', float64),
    ('origin', float64[:]),
    ('dchi', float64),
    ('permeation_probability', float64),
    ('normal_vector', float64[:]),
    ('B0_projection_vector', float64[:])
]

@jitclass(spec_infinite_cylinder_2D)
class InfiniteCylinder2DNumba:
              
    def __init__(
        self, 
        label, 
        diameter, 
        B0_theta, 
        B0_phi, 
        origin,
        dchi, 
        permeation_probability
    ):

        #initializing the vessel parameters to the vessel object
        self.label = label
        self.diameter = diameter
        self.theta = B0_theta
        self.phi = B0_phi
        self.origin = origin
        self.dchi = dchi
        self.permeation_probability = permeation_probability
        
        self.B0_projection_vector = np.array(
            [
                np.sin(self.theta)*np.cos(self.phi),
                np.sin(self.theta)*np.sin(self.phi)
            ]
        )

        # causes errors when projection vector is [0,0,0], we can fix it without lost of accuracy by doing the following
        if np.linalg.norm(self.B0_projection_vector) == 0:
            self.B0_projection_vector = np.array([0.0,1.0])


    def is_IV_dBz(self, positions, B0):
        radial_vectors=positions-self.origin
        radial_distances= np.sqrt(
            radial_vectors[:, 0]**2 + \
            radial_vectors[:, 1]**2
        )
        
        #check if the point(s) are inside the vessel using the cylinder equation
        is_IV = radial_distances<=(self.diameter/2)
        
        cos_phis = np.dot(
            radial_vectors, 
            np.ascontiguousarray(self.B0_projection_vector)) / (radial_distances * np.linalg.norm(self.B0_projection_vector)
        )

        cos_2phis = 2 * cos_phis**2 - 1
        
        dBz_EV = B0*2*np.pi*self.dchi*((0.5*self.diameter/radial_distances)**2)*cos_2phis*(np.sin(self.theta))**2  
        dBz_IV = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1) 

        # some vessel intersection may cause spins to use EV values for dBz even when they are IV
        # this makes all IV values in the dBz_EV equal to the value at the vessel boundary where
        # "0.5*self.diam == radial_distance_magnitudes"
        dBz_EV[is_IV] = B0*2*np.pi*self.dchi * \
            cos_2phis[is_IV]*(np.sin(self.theta))**2

        return is_IV, dBz_EV, dBz_IV

    def is_IV(self, positions):

        xy=positions-self.origin
        
        x=xy[:,0]
        y=xy[:,1]   
        
        #check if the point(s) are inside the vessel using the cylinder equation
        is_IV = x**2 + y**2 <= (self.diameter/2)**2
        
        return is_IV

    def grid_is_IV(self, N, subvox_size):

        grid_origin = self.origin / subvox_size + (N/2)

        X = np.reshape(np.arange(N), (N,1)) - grid_origin[0]
        Y = np.reshape(np.arange(N), (1,N)) - grid_origin[1]

        grid_radius = (self.diameter/2) / subvox_size

        is_IV = X**2 + Y**2 <= grid_radius**2

        return is_IV

    def dBz_EV(self, positions, B0):

        #finding the phi angle and r distance for all points
        radial_distances, cos_2phis = self._radial_distances_and_angles_from_positions(positions)

        is_IV = radial_distances <= self.diameter/2

        #calculating field offset using the appropriate equation (dChi in cgs, angles in radians, lengths in mm)
        dBz = B0*2*np.pi*self.dchi*((0.5*self.diameter/radial_distances)**2)*cos_2phis*(np.sin(self.theta))**2
        
        # some vessel intersection may cause spins to use EV values for dBz even when they are IV
        # this makes all IV values in the dBzEV equal to the value at the vessel boundary where
        # "0.5*self.diam == radial_distance_magnitudes"
        dBz[is_IV] = B0*2*np.pi*self.dchi * \
            cos_2phis[is_IV]*(np.sin(self.theta))**2

        return dBz
    
    def dBz_IV(self, B0):
        
        dBz = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1)
        
        return dBz

    def intersects(self, other):

        # vector between each vessel's origin
        xy = other.origin - self.origin
        
        x = xy[:,0]
        y = xy[:,1]

        # minimum distance to allowed before intersection
        min_distance = (self.diameter + other.diameter) / 2

        # if the vector distance is less than the minimum distance, then there is intersection
        intersects = x**2 + y**2 <= min_distance**2
        
        return intersects
    
    def volume_fraction(self, voxel_size: float) -> float:
        total_volume = voxel_size**2
        vessel_volume = np.pi*(self.diameter/2)**2
        
        volume_fraction = vessel_volume/total_volume

        return volume_fraction

    def _radial_distances_and_angles_from_positions(self, points):

        #finding the distance between the central axis of the vessel and the point
        radial_vectors=self.origin-points
        
        radial_distances= np.sqrt(
            radial_vectors[:, 0]**2 + \
            radial_vectors[:, 1]**2
        )
        
        cos_phis = np.dot(radial_vectors, np.ascontiguousarray(self.B0_projection_vector)) / (radial_distances * np.linalg.norm(self.B0_projection_vector))
        cos_2phis = 2 * cos_phis**2 - 1
        
        return (radial_distances, cos_2phis)  

class InfiniteCylinder2D:

    def __new__(
        cls,
        diameter: float,
        B0_theta: float, 
        B0_phi: float, 
        origin: np.ndarray, 
        dchi: float, 
        permeation_probability: float=0,
        label: str='',
    ):
    
        return InfiniteCylinder2DNumba(
            label=label,
            diameter=diameter,
            B0_theta=B0_theta, 
            B0_phi=B0_phi, 
            origin=origin,
            dchi=dchi, 
            permeation_probability=permeation_probability 
        )

    @staticmethod
    def from_random(
        diameter: float, 
        dchi: float,
        voxel_size: float,
        permeation_probability: float=0,
        label: str='',
        rng: np.random.Generator = np.random.default_rng()
    ) -> InfiniteCylinder3DNumba:

        origin = (rng.random(2)-0.5)*voxel_size
        
        B0_theta=np.arccos(2*rng.random()-1)
        B0_phi=2*np.pi*rng.random()
        
        #create a new vessel object using the generated parameters and add it to the vessel list
        return InfiniteCylinder2DNumba(
            label,
            diameter,
            B0_theta,
            B0_phi,
            origin,
            dchi,
            permeation_probability,
        )

spec_sphere_3D = [
    ('label', types.unicode_type),
    ('diameter', float64),
    ('origin', float64[:]),
    ('dchi', float64),
    ('permeation_probability', float64)
]

@jitclass(spec_sphere_3D)
class Sphere3DNumba:

    def __init__(self,
        label,
        diameter,
        origin,
        dchi,
        permeation_probability=0.0
    ):
        """Object containing all parameters for a sphere vessel

        Parameters
        ----------
        label : uint8[:]
            uft-8 encoded string, to identify the vessel.
        diameter : float64
            vessel diameter (mm)
        origin : float64[:]
            cartesian coordinates of the vessel origin (mm)
        dchi : float64
            susceptibility difference between the vessel and the surrounding tissue (cgs units)
        permeation_probability : float64
            probability for a spin to permeate through the vesel wall (fraction of 1)
        """

        # initializing the vessel parameters to the vessel object
        self.label = label
        self.diameter = diameter
        self.origin = origin
        self.dchi = dchi
        self.permeation_probability = permeation_probability  # for compatibility

    def is_IV_dBz(self, positions, B0):
        # finding the phi angle and r distance for all points
        radial_distances, cos_thetas = self._radial_distances_and_angles(positions)

        # calculating field offset using the appropriate equation (dChi in cgs, angles in radians, lengths in mm)
        dBz_EV = B0*4/3*np.pi*self.dchi * \
            ((0.5*self.diameter/radial_distances)**3)*(3*cos_thetas**2-1)

        is_IV = radial_distances < self.diameter/2

        return is_IV, dBz_EV, 0

    def is_IV(self, positions):
        positions0 = positions-self.origin

        # check if the point(s) are inside the vessle using the cylinder equation
        is_IV = positions0[:, 0]**2 + positions0[:, 1]**2 + positions0[:, 2]**2 <= (self.diameter/2)**2

        return is_IV

    def grid_is_IV(self, N, subvox_size):
        grid_origin = self.origin / subvox_size + (N/2)

        X = np.reshape(np.arange(N), (N,1,1)) - grid_origin[0]
        Y = np.reshape(np.arange(N), (1,N,1)) - grid_origin[1]
        Z = np.reshape(np.arange(N), (1,1,N)) - grid_origin[2]

        grid_radius = (self.diameter/2) / subvox_size

        is_IV = X**2 + Y**2 + Z**2 <= grid_radius**2

        return is_IV

    def dBz_EV(self, positions, B0):

        # finding the phi angle and r distance for all points
        radial_distances, cos_thetas = self._radial_distances_and_angles(positions)

        # calculating field offset using the appropriate equation (dChi in cgs, angles in radians, lengths in mm)
        dBz = B0*4/3*np.pi*self.dchi*((0.5*self.diameter/radial_distances)**3)*(3*cos_thetas**2-1)

        return dBz

    def dBz_IV(self, B0):
        # for compatibility
        return 0

    def intersects(self, other):

        vector = self.origin - other.origin
        distance = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

        intersects = distance < (self.diameter + other.diameter)/2

        return intersects
    
    def volume_fraction(self, voxel_size: float) -> float:
        # calculate the estimated volume percent of the generated vessel
        total_volume = voxel_size**3
        sphere_volume = 4 / 3 * np.pi * (self.diameter / 2)**3
        volume_fraction = sphere_volume / total_volume

        return volume_fraction

    def _radial_distances_and_angles(self, positions):
        # finding the distance between the center of the vessel and the point
        radial_vectors = positions-self.origin
        z = radial_vectors[:, 2]
        radial_distances = np.sqrt(radial_vectors[:, 0]**2 + radial_vectors[:, 1]**2 + radial_vectors[:, 2]**2)

        cos_thetas = z/radial_distances
        return radial_distances, cos_thetas

class Sphere3D:
    """Object containing all parameters for an infinite cylinder vessel

    Parameters
    ----------
    label : str
        string to identify the vessel.
    diameter : float
        vessel diameter (mm)
    origin : np.ndarray
        cartesian coordinates of the vessel origin (mm)
    dchi : float
        susceptibility difference between the vessel and the surrounding tissue (cgs units)
    permeation_probability : float
        probability for a spin to permeate through the vesel wall (fraction of 1)
    """
    def __new__(
        cls,
        diameter: float,
        origin: np.ndarray, 
        dchi: float, 
        permeation_probability: float=0,
        label: str=''
    ):
    
        return Sphere3DNumba(
            label=label,
            diameter=diameter,
            origin=origin,
            dchi=dchi, 
            permeation_probability=permeation_probability 
        )

    @staticmethod
    def from_random(
        diameter: float, 
        dchi: float,
        voxel_size: float,
        permeation_probability: float=0,
        label: str='',
        rng: np.random.Generator = np.random.default_rng()
    ) -> InfiniteCylinder3DNumba:

        # generate a random point in the voxel
        origin = (rng.random(3) - 0.5) * voxel_size

        # return vessel object with the generated components
        return Sphere3DNumba(
            label,
            diameter,
            origin,
            dchi,
            permeation_probability
        )

Vessel3D = Union[InfiniteCylinder3DNumba, Sphere3DNumba]
Vessel2D = InfiniteCylinder2DNumba
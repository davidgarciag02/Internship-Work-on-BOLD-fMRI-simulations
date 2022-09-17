import numpy as np
from numba import float64, uint8   # import the types
from numba.experimental import jitclass
from typing import Union

################################# Infinite Cylinder #################################

spec_infinite_cylinder_3D = [
    ('identifier', uint8[:]),
    ('diameter', float64),
    ('theta', float64),
    ('phi', float64),
    ('origin', float64[:]),
    ('volume_percent', float64),
    ('dchi', float64),
    ('permeation_probability', float64),
    ('normal_vector', float64[:]),
    ('B0_projection_vector', float64[:])
]

@jitclass(spec_infinite_cylinder_3D)
class InfiniteCylinder3D:
    """3D infinite cylinder object
    """    

    def __init__(
        self,
        identifier,
        diameter,
        theta, 
        phi, 
        origin, 
        volume_percent, 
        dchi, 
        permeation_probability
    ):
        """Object containing all parameters for an infinite cylinder vessel

        Parameters
        ----------
        identifier : uint8[:]
            uft-8 encoded string, to identify the vessel.
        diameter : float64
            vessel diameter (mm)
        theta : float64
            zenith angle of the vessel direction (radians)
        phi : float64
            azimuth angle of the vessel direction (radians)
        origin : float64[:]
            cartesian coordinates of the vessel origin (mm)
        volume_percent : float64
            volume contribution of the vessel (fraction of 1)
        dchi : float64
            susceptibility difference between the vessel and the surrounding tissue (cgs units)
        permeation_probability : float64
            probability for a spin to permeate through the vesel wall (fraction of 1)
        """

        # initializing the vessel parameters to the vessel object
        self.identifier = identifier
        self.diameter = diameter
        self.theta = theta
        self.phi = phi
        self.origin = origin
        self.volume_percent = volume_percent
        self.dchi = dchi
        self.permeation_probability = permeation_probability
        self.normal_vector = np.array(
            [
                np.sin(self.theta)*np.cos(self.phi),
                np.sin(self.theta)*np.sin(self.phi),
                np.cos(self.theta)
            ]
        )

        B0_direction = np.array([0, 0, 1])
        self.B0_projection_vector = B0_direction-self.normal_vector[2]*self.normal_vector

    def dBz_mask_from_positions(self, positions, B0):

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
        dBz_EV = B0*2*np.pi*self.dchi * \
            (0.5*self.diameter/radial_distances)**2 * \
            cos2_phis*(np.sin(self.theta))**2
        
        #TODO add this everywhere
        # some vessel intersection may cause spins to use EV values for dBz even when they are IV
        # this makes all IV values in the dBzEV equal to the value at the vessel boundary where
        # "0.5*self.diameter == radial_distance_magnitudes"
        dBz_EV[is_IV] = B0*2*np.pi*self.dchi * \
            cos2_phis[is_IV]*(np.sin(self.theta))**2

        dBz_IV = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1)

        return is_IV, dBz_EV, dBz_IV

    def is_IV(self, positions):

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

    def dBz_EV(self, positions, B0):

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

    def dBz_IV(self, B0):
        dBz = B0*4/6*np.pi*self.dchi*(3*np.cos(self.theta)**2-1)
        return dBz

    def intersects(self, other):

        #calculating the shortest distance between the vessel axes
        cross = np.cross(self.normal_vector, other.normal_vector)
        top = np.dot(cross, other.origin - self.origin)
        
        min_distance = np.abs(top) / \
            np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        #checking for intersection
        intersects = min_distance < (self.diameter + other.diameter)/2 
        
        return intersects

    def _radial_vectors_from_positions(self, positions):

        dot_product = np.dot(np.ascontiguousarray(positions),
                            np.ascontiguousarray(self.normal_vector))

        projected_positions = positions - np.expand_dims(dot_product, axis=1) * self.normal_vector
        radial_vectors = self.origin - projected_positions
        return radial_vectors

spec_infinite_cylinder_2D = [
    ('identifier', uint8[:]),
    ('diameter', float64),
    ('theta', float64),
    ('phi', float64),
    ('origin', float64[:]),
    ('volume_percent', float64),
    ('dchi', float64),
    ('permeation_probability', float64),
    ('normal_vector', float64[:]),
    ('B0_projection_vector', float64[:])
]

@jitclass(spec_infinite_cylinder_2D)
class InfiniteCylinder2D:
              
    def __init__(
        self, 
        identifier, 
        diameter, 
        B0_theta, 
        B0_phi, 
        origin, 
        volume_percent, 
        dchi, 
        permeation_probability
    ):

        #initializing the vessel parameters to the vessel object
        self.identifier = identifier
        self.diameter = diameter
        self.theta = B0_theta
        self.phi = B0_phi
        self.origin = origin
        self.volume_percent = volume_percent
        self.dchi = dchi
        self.permeation_probability = permeation_probability
        self.B0_projection_vector = np.array(
            [
                np.sin(self.theta)*np.cos(self.phi),
                np.sin(self.theta)*np.sin(self.phi)
            ]
        )


    def dBz_mask_from_positions(self, positions, B0):
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



################################# Sphere #################################
spec_sphere_3D = [
    ('identifier', uint8[:]),
    ('diameter', float64),
    ('origin', float64[:]),
    ('volume_percent', float64),
    ('dchi', float64),
    ('permeation_probability', float64)
]

@jitclass(spec_sphere_3D)
class Sphere3D:

    def __init__(self,
        identifier,
        diamerter,
        origin, 
        volume_percent,
        dchi,
        permeation_probability=0.0
    ):
        """Object containing all parameters for a sphere vessel

        Parameters
        ----------
        identifier : uint8[:]
            uft-8 encoded string, to identify the vessel.
        diameter : float64
            vessel diameter (mm)
        origin : float64[:]
            cartesian coordinates of the vessel origin (mm)
        volume_percent : float64
            volume contribution of the vessel (fraction of 1)
        dchi : float64
            susceptibility difference between the vessel and the surrounding tissue (cgs units)
        permeation_probability : float64
            probability for a spin to permeate through the vesel wall (fraction of 1)
        """

        # initializing the vessel parameters to the vessel object
        self.identifier = identifier
        self.diameter = diamerter
        self.origin = origin
        self.volume_percent = volume_percent 
        self.dchi = dchi
        self.permeation_probability = permeation_probability  # for compatibility

    def dBz_mask_from_positions(self, positions, B0):
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

    def _radial_distances_and_angles(self, positions):
        # finding the distance between the center of the vessel and the point
        radial_vectors = positions-self.origin
        z = radial_vectors[:, 2]
        radial_distances = np.sqrt(radial_vectors[:, 0]**2 + radial_vectors[:, 1]**2 + radial_vectors[:, 2]**2)

        cos_thetas = z/radial_distances
        return radial_distances, cos_thetas


Vessel3D = Union[InfiniteCylinder3D, Sphere3D]
Vessel2D = InfiniteCylinder2D
#Importing relevant packages
from BOLDswimsuite import BOLDsimulation, BOLDparameters
import numpy as np
import matplotlib.pyplot as plt


def main():

    #initializing the parameters object
    myParameters = BOLDparameters.ParametersDiscrete2D(

        # VOXEL PARAMETERS
        vessel_type='cylinder',      #shape of the vessel
        size=None,                  #voxel edge length (mm), calculated from optional parameter <k> value if set to None
        CBV=0.02,                   #cerebral blood volume (% ratio)

        # VESSEL PARAMETERS
        identifiers =                       ['identifier1'],            #identifiers of vessels, simply used as keys to dictionaries                        

        id_weights =                        {'identifier1': 1},         #weight of each vessel id for the total CBV                      

        id_diameters =                      {'identifier1': [0.002]},    #vessel diameter distribution of each id (mm)                                    

        id_dchis =                          {'identifier1': 3e-8},      #susceptibility difference (vessel to tissue) of each id in cgs                    

        id_permeation_probabilities =       {'identifier1': 0},         #permeation probability of each id

        # SPIN PARAMETERS
        ADC=0.001,          #diffusion length (mm^2/s)
        B0=3,               #magnetic field strength (T)
        dt=0.2,             #time step (ms)
        num_dt=400,          #number of time steps
        num_spins=10_000,      #number of spins
        
        #GRID PARAMETERS
        N=400,              #number of grid points on which the spins will sample offset and vessel mask
        padding=200,          #zero padding added during FFT convolution (not used otherwise) 

        # SIGNAL PARAMETERS
        pulse_time_indices=[0, 175],                                 #pulse time indices (first pulse always at 0)
        pulse_angles=[np.pi/2, np.pi],                       #angle of each pulse (radians)
        pulse_axes=[[np.pi/2, np.pi/2],[np.pi/2, 0]],        #pulse axes in polar coordinates (radians) (phi,theta)
        
        # OPTIONAL PARAMETERS
        num_vessels=20,      #number of vessels
        edge_width=0,        #sampling region border width (as a fraction of the space - must be < 0.5)
        T1IV=None,          #intravascular T1 value in ms, ignored if set to None
        T2IV=None,          #intravascular T2 value in ms, ignored if set to None
        T1EV=None,          #extravascular T1 value in ms, ignored if set to None
        T2EV=None,          #extravascular T2 value in ms, ignored if set to None
        dB1fact=1,          #dB1 offset as a fraction of the dephasing angle
        dB0=0               #dB0 offset in T
    )
    
    #initializing the simulation object
    mySimulation = BOLDsimulation.SimulationDiscrete2D(myParameters, IV=True, voxelseed=None, spinseed=None) #when IV set to True, will include intravascular effects
    
    print('grid points per diameter:', myParameters.N / myParameters.size * 0.002)

    #printing number of vessels
    print(len(mySimulation.voxel.vessels))
    #printing voxel size
    print(myParameters.size)

    #starting the simulation (enabling the progress bar)
    signal, time_range, sim_time = mySimulation.signal_simulation(
        progressbar=True, 
        offset_method='AnalyticalFromVoxel', 
        diffusion_scheme='Deterministic',
        kernel_type='ModifiedBessel',
        permeable=True,
        extend=True
    ) #progressbar toggles the terminal progress monitoring
    
    #printing the simulation time
    print('\n Total Elapsed Time: ', sim_time)
    #plotting results

    f, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    ax1.plot(time_range, signal[0])
    ax1.set_title('Total')
    ax2.plot(time_range, signal[1])
    ax2.set_title('EV')
    ax3.plot(time_range, signal[2])
    ax3.set_title('IV')
    f.tight_layout()
    plt.show()
    
 
if __name__ == "__main__":
    main()
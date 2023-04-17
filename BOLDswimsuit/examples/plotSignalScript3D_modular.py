#Importing relevant packages
from BOLDswimsuite import BOLDvoxel, BOLDspins, BOLDsequence
import numpy as np
import matplotlib.pyplot as plt

def main():

    vessel_diameter = 0.002
    size = 20 * vessel_diameter
    
    voxel = BOLDvoxel.Voxel3D.from_random(
        size=size,
        CBV=0.02,
        identifiers=['vsl1'],
        id_weights={'vsl1':1},
        id_diameters={'vsl1':[vessel_diameter]},
        id_dchis={'vsl1':3e-8},
        id_permeation_probabilities={'vsl1':0},
        vessel_type='cylinder',
        allow_vessel_intersection=True,
        seed=None,
        progressbar=True
    )
        
    spins = BOLDspins.SpinsContinuous3D(
        ADC=0.001,
        B0=3,
        num_spins=10_000,
        num_dt=600,
        dt=0.2,
        edge_width=0,
        seed=None
    )

    spins.random_walk(
        voxel=voxel, 
        IV=True,
        progressbar=True
    )

    sequence = BOLDsequence.Sequence(
        pulse_time_indices=[0,175],
        pulse_angles=[np.pi/2,np.pi],
        pulse_axes=[[np.pi/2, np.pi/2], [np.pi/2, 0]],
        dB0=0,
        dB1fact=1,
        T2EV=None,
        T2IV=None,
        T1EV=None,
        T1IV=None           
    )

    signal = sequence.signal_from_spins(
        spins=spins,
        cplx=False
    )

    time_range = np.arange(0, spins.num_dt * spins.dt + spins.dt, spins.dt)

    #printing number of vessels
    print(len(voxel.vessels))

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
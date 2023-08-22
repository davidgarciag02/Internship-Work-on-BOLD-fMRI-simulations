#Importing relevant packages
from BOLDswimsuite import BOLDgeometry, BOLDsequence, BOLDspins
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():

    vessel_diameter = 0.2
    nsteps = 600

    size = BOLDgeometry.size_from_k(
        diameter=vessel_diameter, 
        k=20,
        ADC=0.001,
        dt=0.2
    )
    
    voxel = BOLDgeometry.ContinuousVoxel3D.from_random(
        size=size,
        CBV=0.02,
        B0=3,
        labels=['vsl1'],
        id_weights={'vsl1':1},
        id_diameters={'vsl1':[vessel_diameter]},
        id_dchis={'vsl1':3e-8},
        id_permeation_probabilities={'vsl1':0},
        vessel_type='cylinder',
        allow_vessel_intersection=True,
        seed=1,
        progressbar=True
    )

    # grid = BOLDgeometry.DiscreteVoxel3D.from_continuous_analytical(
    #     N=400,
    #     voxel=voxel
    # )
        
    spins = BOLDspins.Spins3D(
        ADC=0.001,
        num_spins=1_000,
        geometry=voxel,
        dt=0.2,
        IV=True,
        seed=1
    )

    sequence = BOLDsequence.Sequence(
        num_samples=spins.num_spins,
        pulse_time_indices=[0,175],
        pulse_angles=[np.pi/2,np.pi],
        pulse_axes=[[np.pi/2, np.pi/2], [np.pi/2, 0]],
        dB0=0          
    )

    eviv = np.zeros(nsteps)
    ev = np.zeros(nsteps)
    iv = np.zeros(nsteps)

    for i in tqdm(range(nsteps)):
        
        spins.step(dt=0.2)
        sequence.spins_step(spins=spins)
        eviv[i], ev[i], iv[i] = sequence.get_signals()

    time_range = np.arange(0, nsteps * spins.dt, spins.dt)

    #printing number of vessels
    print('Number of vessels:', len(voxel.vessels))

    #plotting
    f, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    ax1.plot(time_range, eviv)
    ax1.set_title('Total')
    ax2.plot(time_range, ev)
    ax2.set_title('EV')
    ax3.plot(time_range, iv)
    ax3.set_title('IV')
    f.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
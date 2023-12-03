import numpy as np
from BOLDswimsuite import BOLDgeometry

size = 2

voxel1 = BOLDgeometry.ContinuousVoxel3D.from_random(
    size=size,
    CBV=0.01,
    B0=3,
    labels=['1'],
    weights={'1':1},
    diameter_distributions={'1': np.linspace(0.01, 0.05, 10)},
    dchis={'1':3e-8},
    seed=1
)
print("Cylinders: ", len(voxel1.vessels))

voxel2 = BOLDgeometry.ContinuousVoxel3D.from_random(
    size=size,
    CBV=0.01,
    B0=3,
    labels=['1'],
    weights={'1':1},
    diameter_distributions={'1': np.linspace(0.02, 0.1, 10)},
    dchis={'1':3e-8},
    vessel_type='sphere',
    seed=1
)

print("Spheres: ", len(voxel2.vessels))

voxel = BOLDgeometry.ContinuousVoxel3D(
    size=size,
    B0=3,
    vessels=voxel1.vessels + voxel2.vessels
)

voxel.show()
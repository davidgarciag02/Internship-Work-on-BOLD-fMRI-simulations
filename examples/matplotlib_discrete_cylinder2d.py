from BOLDswimsuite import BOLDgeometry
import matplotlib.pyplot as plt
import numpy as np

size = 2

voxel = BOLDgeometry.ContinuousVoxel2D.from_random(
    size=size,
    CBV=0.03,
    B0=3,
    labels=['1'],
    weights={'1': 1},
    diameter_distributions={'1': [0.01, 0.02, 0.01, 0.02, 0.03, 0.05, 0.06]},
    dchis={'1': 3e-8},
    seed=1,
    vessel_type='cylinder'
)

print(len(voxel.vessels))

voxel.show()
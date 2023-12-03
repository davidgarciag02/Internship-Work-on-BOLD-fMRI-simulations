from BOLDswimsuite import BOLDgeometry

size = 2

voxel = BOLDgeometry.ContinuousVoxel3D.from_random(
    size=size,
    CBV=0.01,
    B0=3,
    labels=['1'],
    weights={'1': 1},
    diameter_distributions={'1': [0.05]},
    dchis={'1': 3e-8},
    seed=1,
    vessel_type='sphere'
)

voxel.show()
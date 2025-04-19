import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kalman_filter(data: np.ndarray) -> np.ndarray:
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data[0]], [0]])
    kf.F = np.array([[1, 1], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 1000
    kf.R = 5
    kf.Q = np.array([[0.01, 0.01], [0.01, 0.1]])
    filtered = []
    for z in data:
        kf.predict()
        kf.update(z)
        filtered.append(kf.x[0,0])
    return np.round(filtered, 2)

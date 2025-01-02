import numpy as np
import jax.numpy as jnp
from manifpy import SE3, SE3Tangent, SO3, SO3Tangent
from scipy.spatial.transform import Rotation
import numpy as np
import sys
sys.path.append('..')
sys.path.append('.')
from traoptlibrary.traopt_utilis import se3_vee
import matplotlib.pyplot as plt

p0 = np.array([1., 1., -1.])
quat_ref = Rotation.from_euler(
        'zyx', [ 0., 30., 0. ], degrees=True
        ).as_quat()
q_ref_mnf = SE3( position=p0, quaternion=quat_ref )
angle_range = np.arange(-360., 360.5, 0.5)

errs_left = []
errs_right = []
errs_cmptb = []

for theta in angle_range:
    quat0 = Rotation.from_euler(
        'zyx', [ theta, 0., 0. ], degrees=True
        ).as_quat()
    q_mnf = SE3( position=p0, quaternion=quat0 )
    errs_left.append( q_mnf.lminus( q_ref_mnf ) )
    errs_right.append( q_mnf.rminus( q_ref_mnf ) )
    errs_cmptb.append(0.5*se3_vee(
            (q_mnf.inverse() * q_ref_mnf).transform() - (q_ref_mnf.inverse() * q_mnf).transform()
        ))
    
# errs_left_norm = [ np.linalg.sqrt( x.coeffs().T @ x.coeffs() ) for x in errs_left]
errs_left_norm = [ np.linalg.norm( x.coeffs() ) for x in errs_left]
errs_right_norm = [ np.linalg.norm( x.coeffs() ) for x in errs_right]
errs_cmptb_norm = [ np.linalg.norm( x ) for x in errs_cmptb]

k = 100
print("for",k,"-th element\n",
      "left-error is\n", errs_left[k],
      "\nleft-error norm is\n", errs_left_norm[k],
      "\nright-error is\n", errs_right[k],
      "\nright-error norm is\n", errs_right_norm[k])

plt.figure()
plt.plot( angle_range, errs_left_norm )
plt.plot( angle_range, errs_right_norm )
plt.plot( angle_range, errs_cmptb_norm )
plt.xlabel("Angle/degree")
plt.legend(["Left","Right","Compatible"])
plt.title("Left/Right Cost Comparison")
plt.grid()
plt.show()
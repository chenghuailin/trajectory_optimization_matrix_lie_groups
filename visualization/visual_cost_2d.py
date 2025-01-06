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
angle_ref_z = 30.
quat_ref = Rotation.from_euler(
        'zyx', [ angle_ref_z, 10., 0. ], degrees=True
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



# plt.figure()
# plt.plot( angle_range, errs_left_norm )
# plt.plot( angle_range, errs_right_norm )
# plt.plot( angle_range, errs_cmptb_norm )
# plt.axvline( angle_ref_z, linestyle='--' )
# plt.axvline( angle_ref_z-180.,  linestyle='--' )
# plt.axvline( angle_ref_z+180.,  linestyle='--' )
# plt.xlabel("Z-Axis Angle/degree")
# # plt.legend(["Left","Right","Compatible",r'$30^{\circ}$',r'$-150^{\circ}$',r'$210^{\circ}$'])
# plt.legend(["Left","Right","Compatible"])
# plt.title("Difference Norm Visualization")
# plt.grid()
# plt.show()


plt.figure(figsize=(12, 7))
plt.plot(angle_range, errs_left_norm, label=r"$||R\ominus^l R_{ref}||_2$")
plt.plot(angle_range, errs_right_norm, label="$||R\ominus^r R_{ref}||_2$")
# plt.plot(angle_range, errs_cmptb_norm, label="Compatible")

# 添加特殊标注 30°
plt.axvline(30, linestyle='--', color='r')
plt.text(30, plt.ylim()[1]*0.985, '30°', rotation=90, color='r',
         verticalalignment='top', horizontalalignment='right', fontsize=9)

# 添加特殊标注 -150°
plt.axvline(-150, linestyle='--', color='r')
plt.text(-150, plt.ylim()[1]*0.985, '-150°', rotation=90, color='r',
         verticalalignment='top', horizontalalignment='right', fontsize=9)

# 添加特殊标注 210°
plt.axvline(210, linestyle='--', color='r')
plt.text(210, plt.ylim()[1]*0.985, '210°', rotation=90, color='r',
         verticalalignment='top', horizontalalignment='right', fontsize=9)

# # 获取当前 x 轴的刻度
# current_ticks = plt.xticks()[0].tolist()

# # 添加 30°, -150°, 210° 到刻度
# additional_ticks = [30, -150, 210]
# for tick in additional_ticks:
#     if tick not in current_ticks:
#         current_ticks.append(tick)

# current_ticks = sorted(current_ticks)
# plt.xticks(current_ticks)

plt.xlabel(r"Z-axis euler angle $\theta_z$ ( ${}^{\circ}$ )")
# plt.title("Difference Norm Visualization")
plt.grid(True)

plt.legend()
plt.show()
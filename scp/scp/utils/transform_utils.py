# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Rigid-body transformations including velocities and static forces."""


from absl import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from numba import njit
# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = (1.0 - 1e-6)
_TOL = 1e-10


def _clip_within_precision(number, low, high, precision=_TOL):
  """Clips input to provided range, checking precision.

  Args:
    number: (float) number to be clipped.
    low: (float) lower bound.
    high: (float) upper bound.
    precision: (float) tolerance.

  Returns:
    Input clipped to given range.

  Raises:
    ValueError: If number is outside given range by more than given precision.
  """
  if (number < low - precision).any() or (number > high + precision).any():
    raise ValueError(
        'Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}'.
        format(number, low, high, precision))

  return np.clip(number, low, high)


def _batch_mm(m1, m2):
  """Batch matrix multiply.

  Args:
    m1: input lhs matrix with shape (batch, n, m).
    m2: input rhs matrix with shape (batch, m, o).

  Returns:
    product matrix with shape (batch, n, o).
  """
  return np.einsum('bij,bjk->bik', m1, m2)


def _rmat_to_euler_xyz(rmat):
  """Converts a 3x3 rotation matrix to XYZ euler angles."""
  # | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  # | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  # | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  if rmat[0, 2] > _POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at North Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = np.pi/2
    x = 0.0
    return np.array([x, y, z])

  if rmat[0, 2] < -_POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at South Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = -np.pi/2
    x = 0.0
    return np.array([x, y, z])

  z = -np.arctan2(rmat[0, 1], rmat[0, 0])
  y = np.arcsin(rmat[0, 2])
  x = -np.arctan2(rmat[1, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([x, y, z])


def _rmat_to_euler_xyx(rmat):
  """Converts a 3x3 rotation matrix to XYX euler angles."""
  # | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  # | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  # | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |

  if rmat[0, 0] < 1.0:
    if rmat[0, 0] > -1.0:
      y = np.arccos(_clip_within_precision(rmat[0, 0], -1., 1.))
      x0 = np.arctan2(rmat[1, 0], -rmat[2, 0])
      x1 = np.arctan2(rmat[0, 1], rmat[0, 2])
      return np.array([x0, y, x1])
    else:
      # Not a unique solution:  x1_angle - x0_angle = atan2(-r12,r11)
      y = np.pi
      x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
      x1 = 0.0
      return np.array([x0, y, x1])
  else:
    # Not a unique solution:  x1_angle + x0_angle = atan2(-r12,r11)
    y = 0.0
    x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
    x1 = 0.0
    return np.array([x0, y, x1])


def _rmat_to_euler_zyx(rmat):
  """Converts a 3x3 rotation matrix to ZYX euler angles."""
  if rmat[2, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = -np.pi/2
    z = 0.0
    return np.array([z, y, x])

  if rmat[2, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = np.pi/2
    z = 0.0
    return np.array([z, y, x])

  x = np.arctan2(rmat[2, 1], rmat[2, 2])
  y = -np.arcsin(rmat[2, 0])
  z = np.arctan2(rmat[1, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([z, y, x])


def _rmat_to_euler_xzy(rmat):
  """Converts a 3x3 rotation matrix to XZY euler angles."""
  if rmat[0, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = -np.pi/2
    x = 0.0
    return np.array([x, z, y])

  if rmat[0, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = np.pi/2
    x = 0.0
    return np.array([x, z, y])

  y = np.arctan2(rmat[0, 2], rmat[0, 0])
  z = -np.arcsin(rmat[0, 1])
  x = np.arctan2(rmat[2, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([x, z, y])


def _rmat_to_euler_yzx(rmat):
  """Converts a 3x3 rotation matrix to YZX euler angles."""
  if rmat[1, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = np.pi/2
    y = 0.0
    return np.array([y, z, x])

  if rmat[1, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = -np.pi/2
    y = 0.0
    return np.array([y, z, x])

  x = -np.arctan2(rmat[1, 2], rmat[1, 1])
  z = np.arcsin(rmat[1, 0])
  y = -np.arctan2(rmat[2, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([y, z, x])


def _rmat_to_euler_zxy(rmat):
  """Converts a 3x3 rotation matrix to ZXY euler angles."""
  if rmat[2, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = np.pi/2
    z = 0.0
    return np.array([z, x, y])

  if rmat[2, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = -np.pi/2
    z = 0.0
    return np.array([z, x, y])

  y = -np.arctan2(rmat[2, 0], rmat[2, 2])
  x = np.arcsin(rmat[2, 1])
  z = -np.arctan2(rmat[0, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([z, x, y])


def _rmat_to_euler_yxz(rmat):
  """Converts a 3x3 rotation matrix to YXZ euler angles."""
  if rmat[1, 2] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = -np.pi/2
    y = 0.0
    return np.array([y, x, z])

  if rmat[1, 2] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = np.pi/2
    y = 0.0
    return np.array([y, x, z])

  z = np.arctan2(rmat[1, 0], rmat[1, 1])
  x = -np.arcsin(rmat[1, 2])
  y = np.arctan2(rmat[0, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([y, x, z])


def _axis_rotation(theta, full):
  """Returns the theta dim, cos and sin, and blank matrix for axis rotation."""
  n = 1 if np.isscalar(theta) else len(theta)
  ct = np.cos(theta)
  st = np.sin(theta)

  if full:
    rmat = np.zeros((n, 4, 4))
    rmat[:, 3, 3] = 1.
  else:
    rmat = np.zeros((n, 3, 3))

  return n, ct, st, rmat

# map from full rotation orderings to euler conversion functions
_eulermap = {
    'XYZ': _rmat_to_euler_xyz,
    'XYX': _rmat_to_euler_xyx,
    'ZYX': _rmat_to_euler_zyx,
    'XZY': _rmat_to_euler_xzy,
    'YZX': _rmat_to_euler_yzx,
    'ZXY': _rmat_to_euler_zxy,
    'YXZ': _rmat_to_euler_yxz
}


def euler_to_quat(euler):
    """
    Converts euler angles into quaternion form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: (w,x,y,z) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    # Convert Euler angles to quaternion
    quat = R.from_euler("XYZ", euler).as_quat()
    
    # Roll the quaternion to change the order to (w,x,y,z)
    return quat


def euler_to_rmat(euler,ordering="XYZ"):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    return R.from_euler(ordering, euler).as_matrix()

def quat_conj(quat):
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return np.stack(
      [quat[..., 0], -quat[..., 1],
       -quat[..., 2], -quat[..., 3]], axis=-1).astype(np.float64)


def quat_inv(quat):
  """Return inverse of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return quat_conj(quat) / np.sum(quat * quat, axis=-1, keepdims=True)


def _get_qmat_indices_and_signs():
  """Precomputes index and sign arrays for constructing `qmat` in `quat_mul`."""
  w, x, y, z = range(4)
  qmat_idx_and_sign = np.array([
      [w, -x, -y, -z],
      [x, w, -z, y],
      [y, z, w, -x],
      [z, -y, x, w],
  ])
  indices = np.abs(qmat_idx_and_sign)
  signs = 2 * (qmat_idx_and_sign >= 0) - 1
  # Prevent array constants from being modified in place.
  indices.flags.writeable = False
  signs.flags.writeable = False
  return indices, signs

_qmat_idx, _qmat_sign = _get_qmat_indices_and_signs()


def quat_mul(quat1, quat2):
  """Computes the Hamilton product of two quaternions.

  Any number of leading batch dimensions is supported.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product quat1 * quat2.
  """
  # Construct a (..., 4, 4) matrix to multiply with quat2 as shown below.
  qmat = quat1[..., _qmat_idx] * _qmat_sign

  # Compute the batched Hamilton product:
  # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
  # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
  # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
  # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
  return (qmat @ quat2[..., None])[..., 0]


def quat_diff(source, target):
  """Computes quaternion difference between source and target quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the rotation from source to target.
  """
  return quat_mul(quat_conj(source), target)


def quat_log(quat, tol=_TOL):
  """Log of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    4D array representing the log of `quat`. This is analogous to
    `rmat_to_axisangle`.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  q_norm = np.linalg.norm(quat + tol, axis=-1, keepdims=True)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  # Clip to 2*tol because we subtract it here
  v_new = v / np.linalg.norm(v + tol, axis=-1, keepdims=True) * np.arccos(
      _clip_within_precision(a - tol, -1., 1., precision=2.*tol)) / q_norm
  return np.stack(
      [np.log(q_norm[..., 0]), v_new[..., 0], v_new[..., 1], v_new[..., 2]],
      axis=-1)


def quat_dist(source, target):
  """Computes distance between source and target quaternions.

  This function assumes that both input arguments are unit quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    Scalar representing the rotational distance from source to target.
  """
  quat_product = quat_mul(source, quat_inv(target))
  quat_product /= np.linalg.norm(quat_product, axis=-1, keepdims=True)
  return np.linalg.norm(quat_log(quat_product), axis=-1, keepdims=True)


def quat_rotate(quat, vec):
  """Rotate a vector by a quaternion.

  Args:
    quat: A quaternion [w, i, j, k].
    vec: A 3-vector representing a position.

  Returns:
    The rotated vector.
  """
  qvec = np.hstack([[0], vec])
  return quat_mul(quat_mul(quat, qvec), quat_conj(quat))[1:]


def quat_to_axisangle(quat):
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
  """
  angle = 2 * np.arccos(_clip_within_precision(quat[0], -1., 1.))

  if angle < _TOL:
    return np.zeros(3)
  else:
    qn = np.sin(angle/2)
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    axis = quat[1:4] / qn
    return axis * angle


# def quat_to_euler(quat, ordering='XYZ'):
#   """Returns the euler angles corresponding to the provided quaternion.

#   Args:
#     quat: A quaternion [w, i, j, k].
#     ordering: (str) Desired euler angle ordering.

#   Returns:
#     euler_vec: The euler angle rotations.
#   """
#   mat = quat_to_mat(quat)
#   return rmat_to_euler(mat[0:3, 0:3], ordering=ordering)
def quat_to_euler(quat, ordering='XYZ'):
    """Returns the euler angles corresponding to the provided quaternion.

    Args:
        quat: A quaternion [w, i, j, k].
        ordering: (str) Desired euler angle ordering.

    Returns:
        euler_vec: The euler angle rotations.
    """
    # 创建旋转对象
    rotation = R.from_quat(quat)  # 转换为 [x, y, z, w] 格式

    # 根据指定的顺序计算欧拉角
    euler_vec = rotation.as_euler(ordering, degrees=False)

    return euler_vec

def quat_to_mat(quat):
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  q = np.array(quat, dtype=np.float64, copy=True)
  nq = np.dot(q, q)
  if nq < _TOL:
    return np.identity(4)
  q *= np.sqrt(2.0 / nq)
  q = np.outer(q, q)
  return np.array(
      ((1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0),
       (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0),
       (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0),
       (0.0, 0.0, 0.0, 1.0)),
      dtype=np.float64)

# def quat_to_rmat(quat):
#   """Return rotation matrix from quaternion.

#   Args:
#     quat: A quaternion [w, i, j, k].

#   Returns:
#     A 3*3 matrix with the rotation corresponding to `quat`.
#   """
#   q = np.array(quat, dtype=np.float64, copy=True)
#   nq = np.dot(q, q)
#   if nq < _TOL:
#     return np.identity(4)
#   q *= np.sqrt(2.0 / nq)
#   q = np.outer(q, q)
  
#   return np.array(
#       ((1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
#        (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
#        (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),),
#       dtype=np.float64)
def quat_to_rmat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (..., 4) (x,y,z,w) float quaternion angles

    Returns:
        np.array: (..., 3, 3) rotation matrix
    """
    return R.from_quat(quaternion).as_matrix()
def rotation_x_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the X-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transform.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.array([[1, 0, 0]])
  rmat[:, 1, 0:3] = np.vstack([np.zeros(n), ct, -st]).T
  rmat[:, 2, 0:3] = np.vstack([np.zeros(n), st, ct]).T

  return rmat.squeeze()


def rotation_y_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the Y-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, np.zeros(n), st]).T
  rmat[:, 1, 0:3] = np.array([[0, 1, 0]])
  rmat[:, 2, 0:3] = np.vstack([-st, np.zeros(n), ct]).T

  return rmat.squeeze()


def rotation_z_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the z-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, -st, np.zeros(n)]).T
  rmat[:, 1, 0:3] = np.vstack([st, ct, np.zeros(n)]).T
  rmat[:, 2, 0:3] = np.array([[0, 0, 1]])

  return rmat.squeeze()


def rmat_to_euler(rmat, ordering='XYZ'):
  """Returns the euler angles corresponding to the provided rotation matrix.

  Args:
    rmat: The rotation matrix.
    ordering: (str) Desired euler angle ordering.

  Returns:
    Euler angles corresponding to the provided rotation matrix.
  """
  return _eulermap[ordering](rmat)

@njit(cache=True, fastmath=True)
def mat_to_quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


# ################
# # 2D Functions #
# ################


def rotation_matrix_2d(theta):
  ct = np.cos(theta)
  st = np.sin(theta)
  return np.array([
      [ct, -st],
      [st, ct]
  ])

@njit(cache=True, fastmath=True)
def quat_slerp_jitted(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    (adapted from deoxys)
    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    EPS = 1e-8
    q0 = quat0 / np.linalg.norm(quat0)
    q1 = quat1 / np.linalg.norm(quat1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if np.abs(np.abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0
    angle = np.arccos(d)
    if np.abs(angle) < EPS:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0

def pose_inv(pose_mat):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose_mat (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose_mat[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose_mat[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def convert_pose_quat2mat(poses_quat):
    """
    Convert poses from quat xyzw to mat format.
    Args:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z, w]
    Returns:
    - poses_mat (np.ndarray): [N, 4, 4]
    """
    batched = poses_quat.ndim == 2
    if not batched:
        poses_quat = poses_quat[None]
    
    N = len(poses_quat)
    poses_mat = np.tile(np.eye(4), (N, 1, 1))
    poses_mat[:, :3, 3] = poses_quat[:, :3]
    
    # Reorder quaternion from xyzw to wxyz for quat_to_rmat
    quats = poses_quat[:, 3:]
    # Assuming quat_to_rmat can handle N quaternions
    rotation_matrices = quat_to_rmat(quats)
    poses_mat[:, :3, :3] = rotation_matrices
    
    if not batched:
        poses_mat = poses_mat[0]
    
    return poses_mat
# def convert_pose_euler2mat(poses_euler):
#     """
#     Convert poses from euler to mat format.
#     Args:
#     - poses_euler (np.ndarray): [N, 6]
#     Returns:
#     - poses_mat (np.ndarray): [N, 4, 4]
#     """
#     batched = poses_euler.ndim == 2
#     if not batched:
#         poses_euler = poses_euler[None]
#     poses_mat = np.eye(4)
#     poses_mat = np.tile(poses_mat, (len(poses_euler), 1, 1))
#     poses_mat[:, :3, 3] = poses_euler[:, :3]
#     for i in range(len(poses_euler)):
#         poses_mat[i, :3, :3] = euler_to_rmat(poses_euler[i, 3:])
#     if not batched:
#         poses_mat = poses_mat[0]
#     return poses_mat
def convert_pose_euler2mat(poses_euler):
    """
    Convert poses from euler to mat format.
    
    Args:
    - poses_euler (np.ndarray): [N, 6] or [6] for a single pose.
    
    Returns:
    - poses_mat (np.ndarray): [N, 4, 4] or [4, 4] for a single pose.
    """
    batched = poses_euler.ndim == 2
    if not batched:
        poses_euler = poses_euler[None]
    # Initialize the output array with identity matrices
    poses_mat = np.eye(4).reshape(1, 4, 4)  # Create a single identity matrix
    poses_mat = np.tile(poses_mat, (len(poses_euler), 1, 1))  # Expand to shape [N, 4, 4]
    # Set the position
    poses_mat[:, :3, 3] = poses_euler[:, :3]  
    # Compute rotation matrices for all poses at once
    poses_mat[:, :3, :3] = euler_to_rmat(poses_euler[:, 3:])  # Assume euler_to_rmat can handle [N, 3]
    if not batched:
        poses_mat = poses_mat[0]
    return poses_mat

def convert_pose_mat2quat(poses_mat):
    """
    Convert poses from mat to quat xyzw format.
    Args:
    - poses_mat (np.ndarray): [N, 4, 4]
    Returns:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z, w]
    """
    batched = poses_mat.ndim == 3
    if not batched:
        poses_mat = poses_mat[None]
    poses_quat = np.empty((len(poses_mat), 7))
    poses_quat[:, :3] = poses_mat[:, :3, 3]
    for i in range(len(poses_mat)):
        poses_quat[i, 3:] = mat_to_quat(poses_mat[i])
    if not batched:
        poses_quat = poses_quat[0]
    return poses_quat

def pose_to_mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=pose[0].dtype)
    homo_pose_mat[:3, :3] = quat_to_rmat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=pose[0].dtype)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat

# def convert_pose_euler2quat(poses_euler):
#     """
#     Convert poses from euler to quat xyzw format.
#     Args:
#     - poses_euler (np.ndarray): [N, 6]
#     Returns:
#     - poses_quat (np.ndarray): [N, 7], [x, y, z, w, x, y, z]
#     """
#     batched = poses_euler.ndim == 2
#     if not batched:
#         poses_euler = poses_euler[None]
#     poses_quat = np.empty((len(poses_euler), 7))
#     poses_quat[:, :3] = poses_euler[:, :3]
#     for i in range(len(poses_euler)):
#         poses_quat[i, 3:] = euler_to_quat(poses_euler[i, 3:])
#     if not batched:
#         poses_quat = poses_quat[0]
#     return poses_quat

def convert_pose_euler2quat(poses_euler):
    """
    Convert poses from euler to quat xyzw format.
    
    Args:
    - poses_euler (np.ndarray): [N, 6]
    
    Returns:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z,w]
    """
    batched = poses_euler.ndim == 2
    if not batched:
        poses_euler = poses_euler[None]
    
    # Prepare output array
    poses_quat = np.empty((len(poses_euler), 7))
    poses_quat[:, :3] = poses_euler[:, :3]
    
    # Convert Euler angles to quaternions for the entire batch
    #poses_quat[:, 3:] = euler_to_quat(poses_euler[:, 3:])
    for i in range(len(poses_euler)):
        poses_quat[i, 3:] = euler_to_quat(poses_euler[i, 3:])
    if not batched:
        poses_quat = poses_quat[0]
    
    return poses_quat
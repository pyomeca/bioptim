import numpy as np

def rotation_matrix_xyz(theta, phi, psi):
    # Rotation autour de X (theta)
    Rx_theta = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Rotation autour de Y (phi)
    Ry_phi = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # Rotation autour de Z (psi)
    Rz_psi = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    # Matrice de rotation finale
    Rfixed = Ry_phi @ Rx_theta @ Rz_psi
    Rmobile = Rx_theta @ Ry_phi @ Rz_psi

    return Rfixed, Rmobile


def rotmat_to_euler_xyz(R):
    # Extraction de beta
    beta = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))

    # Cas particulier : si cos(beta) = 0 (gimbal lock)
    if np.isclose(np.cos(beta), 0):
        alpha = 0
        gamma = np.arctan2(R[0,1], R[1,1])
    else:
        alpha = np.arctan2(R[1,0]/np.cos(beta), R[0,0]/np.cos(beta))
        gamma = np.arctan2(R[2,1]/np.cos(beta), R[2,2]/np.cos(beta))
    # inversion
    alpha_inv = gamma
    beta_inv = beta
    gamma_inv = alpha
    return (alpha, beta, gamma), (alpha_inv, beta_inv, gamma_inv)

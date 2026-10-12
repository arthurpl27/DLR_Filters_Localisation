import numpy as np

def L_periodic(d,h,c,r,b):
    # We construct the periodic central finite-difference Laplacian. Since the points are x_j = (j-1)h for all j, then
    # u_j := u(x_j) = u((j-1)h). As a result, u(x_j+h) = u(jh) = u_{j+1} and u(x_j-h) = u((j-2)h) = u_{j-1}. Overall:
    #
    #                         u_xx(x_j) ≈ (u_{j+1} - 2u_j + u_{j-1}) / h^2
    #
    #                            u_x(x_j) ≈ (u_{j+1} - u_{j-1}) / (2h)
    #
    # with periodic boundary conditions u_{-1} = u_{d-1} and u_d = u_0.

    u_xx = -2 * np.eye(d)
    u_xx += np.eye(d, k=1)  # k=1 for upper-diagonal
    u_xx += np.eye(d, k=-1)  # k=-1 for lower-diagonal
    u_x = np.eye(d, k=1)
    u_x += -np.eye(d, k=-1)

    # Periodicity
    u_xx[0, -1] = 1
    u_xx[-1, 0] = 1
    u_x[0, -1] = -1
    u_x[-1, 0] = 1

    u_xx /= h ** 2
    u_x /= (2*h)
    mathcal_L = c * u_xx - b * u_x + r * np.eye(d)  # discretized operator

    return mathcal_L


def L_neumann(d,h,c,r, b):
    # We construct the periodic central finite-difference Laplacian. Since the points are x_j = (j-1)h for all j, then
    # u_j := u(x_j) = u((j-1)h). As a result, u(x_j+h) = u(jh) = u_{j+1} and u(x_j-h) = u((j-2)h) = u_{j-1}. Overall:
    #
    #                         u_xx(x_j) ≈ (u_{j+1} - 2u_j + u_{j-1}) / h^2
    #
    #                           u_x(x_j) ≈ (u_{j+1} - u_{j-1}) / (2h)
    #
    # with Neumann boundary conditions u_1 = u_{-1} and u_{d+1} = u_{d-1}.

    u_xx = -2 * np.eye(d)
    u_xx += np.eye(d, k=1)  # k=1 for upper-diagonal
    u_xx += np.eye(d, k=-1)  # k=-1 for lower-diagonal

    u_x = np.eye(d, k=1)
    u_x += -np.eye(d, k=-1)

    # Neumann
    u_xx[0, 1] = 2
    u_xx[-1, -2] = 2
    u_x[0, 1] = 0
    u_x[-1, -2] = 0

    u_xx /= h ** 2
    u_x /= (2*h)

    mathcal_L2 = c * u_xx - b * u_x + r * np.eye(d)  # discretized operator

    return mathcal_L2
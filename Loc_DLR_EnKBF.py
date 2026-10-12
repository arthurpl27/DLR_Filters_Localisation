import numpy as np
import matplotlib.pyplot as plt
from PDE_tools import L_periodic, L_neumann
from Localisation_tools import covariance_loc_gc


# ============================================================
#                    PARAMETERS & SEED
# ============================================================

L = 10
d = 100
h = L / d
T = 20
dt = 1e-2
N = int(T / dt)
c = 0.05
b = 0.05
r = 0.02
rank = 25
M = 100
k = 10
sigma = 0.1
gamma = 1
radius_index = 5            # localization radius
radius_phys = radius_index * h

seed = 42
rng = np.random.default_rng(seed)


# ============================================================
#                 SPATIAL GRID & OPERATOR
# ============================================================

x = np.arange(d) * h
time = np.linspace(0, T, N + 1)

I_d = np.eye(d)
I_rank = np.eye(rank)
I_k = np.eye(k)

mathcal_L = L_neumann(d,h,c,r, b)

# ============================================================
#               MODEL & OBSERVATION PROCESSES
# ============================================================

observation_indices = np.linspace(0, d-1, k, dtype=int)
H = np.zeros((k, d))

for i, j in enumerate(observation_indices):
    H[i, j] = 1


Sigma = sigma * I_d
sqrt_Sigma = np.sqrt(sigma) * I_d

Gamma = gamma * I_k
Gamma_inv = (1 / gamma) * I_k
sqrt_Gamma = np.sqrt(gamma) * I_k
Gamma_inv_sqrt = (1 / np.sqrt(gamma)) * I_k

# Initial condition: mean U_0^0
m_dlr = np.cos(2 * np.pi * x / L)

# initial deterministic modes
U_raw = np.zeros((d, rank))
for j in range(1, rank + 1):

    U_raw[:, j-1] = np.cos(2 * np.pi * j * x / L)

# initial stochastic modes
Y_raw = rng.standard_normal((rank, M))

# Initial orthonormalization
U, R_qr = np.linalg.qr(U_raw, mode="reduced")
Y = R_qr @ Y_raw


# ============================================================
#                 INITIAL TRUE SIGNAL
# ============================================================

Y_true_0 = rng.standard_normal(rank)
X_true = m_dlr + U_raw @ Y_true_0

# create arrays
true_signal = np.zeros((N + 1, d))
mean_signal_dlr_loc = np.zeros((N + 1, d))
rmse_values_dlr_loc = np.zeros(N + 1)

true_signal[0] = X_true
mean_signal_dlr_loc[0] = m_dlr
rmse_values_dlr_loc[0] = np.sqrt(np.mean((m_dlr - X_true) ** 2))


# ============================================================
#
#                        DLR - KBF
#
# ============================================================

for n in range(N):

    # true signal
    X_n = X_true.copy()

    # observations
    dB_obs = np.sqrt(dt) * rng.standard_normal(k)
    dZ = H @ X_n * dt + sqrt_Gamma @ dB_obs

    # true signal EM update
    dW_true = np.sqrt(dt) * rng.standard_normal(d)
    X_true = X_n + mathcal_L @ X_n * dt + sqrt_Sigma @ dW_true


    # ========================================================
    #                    PREDICTION STEP
    # ========================================================

    # mean_hat update
    m_hat = np.linalg.solve(I_d - dt * mathcal_L,m_dlr)

    # deterministic_modes_hat update
    projector_perp = I_d - U @ U.T
    lhs_U = I_d - dt * projector_perp @ mathcal_L
    U_tilde = np.linalg.solve(lhs_U, U)

    # stochastic_modes_hat update
    dW = np.sqrt(dt) * rng.standard_normal((d, M))
    L_U = U.T @ mathcal_L @ U
    lhs_Y_prediction = I_rank - dt * L_U
    rhs_Y_prediction = Y + U.T @ sqrt_Sigma @ dW
    Y_tilde = np.linalg.solve(lhs_Y_prediction, rhs_Y_prediction)

    # ========================================================
    #                 QR RE-ORTHONORMALIZATION
    # ========================================================

    U_hat, R_qr = np.linalg.qr(U_tilde, mode="reduced")
    Y_hat = R_qr @ Y_tilde

    # ========================================================
    #               SAMPLE COVARIANCE OF Y
    # ========================================================

    C_Y_hat = (Y_hat @ Y_hat.T) / (M - 1)

    # ========================================================
    #                    ANALYSIS STEP
    # ========================================================

    # Loc-DLR mean update
    matrix1 = U_hat @ Y_hat
    matrix2 = matrix1.T @ H.T
    mathcal_U = (matrix1 @ matrix2) / (M - 1)
    mathcal_U_loc = covariance_loc_gc(radius_phys, mathcal_U, row_positions=x, col_positions=observation_indices)
    mathcal_V = Gamma_inv
    mathcal_VH = mathcal_V @ H

    rhs_mean = m_hat + mathcal_U_loc @ mathcal_V @ dZ
    woodbury_matrix = I_k + mathcal_VH @ mathcal_U_loc * dt
    rhs_solve = mathcal_VH @ rhs_mean
    woodbury_solve = np.linalg.solve(woodbury_matrix, rhs_solve)
    m_new = rhs_mean - mathcal_U_loc @ woodbury_solve * dt

    # Loc-DLR stochastic modes update
    dB = np.sqrt(dt) * rng.standard_normal((k, M))
    lhs_Y_analysis = I_rank + U_hat.T @ mathcal_U_loc @ mathcal_VH @ U_hat * dt
    rhs_Y_analysis = Y_hat - U_hat.T @ mathcal_U_loc @ Gamma_inv_sqrt @ dB
    Y_new = np.linalg.solve(lhs_Y_analysis, rhs_Y_analysis)

    # ========================================================
    #                    UPDATE VARIABLES
    # ========================================================

    m_dlr = m_new
    U = U_hat
    Y = Y_new

    true_signal[n + 1] = X_true
    mean_signal_dlr_loc[n + 1] = m_dlr
    rmse_values_dlr_loc[n + 1] = np.sqrt(np.mean((m_dlr - X_true) ** 2))


# ============================================================
#                 FINAL DLR APPROXIMATION
# ============================================================

# Reconstruct the full DLR ensemble for each particle X_DLR^(p) = m_DLR + U Y^(p)
ensemble_dlr = m_dlr[:, None] + U @ Y

# Stochastic covariance
C_Y_final = (Y @ Y.T) / (M-1)

# Full DLR covariance
C_dlr_final = U @ C_Y_final @ U.T

# ============================================================
#                     INFORMATION
# ============================================================

print("============================================")
print(f"State dimension d:          {d}")
print(f"DLR rank:                   {rank}")
print(f"Observation dimension k:    {k}")
print(f"Number of Y samples M:      {M}")
print("============================================")
print("U shape:", U.shape)
print("Y shape:", Y.shape)
print("DLR covariance shape:", C_dlr_final.shape)
print("Final RMSE:", rmse_values_dlr_loc[-1])


# ============================================================
#                   FILTERED MEAN
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(x, true_signal[-1], label="True signal")
plt.plot(x, mean_signal_dlr_loc[-1], linewidth=2, label="Loc-DLR-KBP mean")
plt.scatter(x[observation_indices], true_signal[-1, observation_indices], marker="x", label="Observed locations")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,T)$")
plt.title(f"Loc-DLR-KBP at T = {T}, rank = {rank}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ============================================================
#                       RMSE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(time, rmse_values_dlr_loc)
plt.xlabel(r"$t$")
plt.ylabel("RMSE")
plt.title("Loc-DLR-KBP estimation error")
plt.grid(alpha=0.3)
plt.show()


# ============================================================
#                  A FEW DLR SAMPLES
# ============================================================

plt.figure(figsize=(10, 5))

number_samples_plot = min(5, M)
for p in range(number_samples_plot):

    plt.plot(x, ensemble_dlr[:, p], alpha=0.6, label=f"Sample {p + 1}")

plt.plot(x, m_dlr, linewidth=3, label="DLR mean")
plt.plot(x, X_true, linestyle="--", linewidth=2, label="True signal")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,T)$")
plt.title(f"Loc-DLR-KBP samples at T = {T}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ============================================================
#                     DLR COVARIANCE
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(C_dlr_final, origin="lower", extent=[0, L, 0, L], aspect="auto")
plt.colorbar(label="Localized Covariance")
plt.xlabel(r"$x_j$")
plt.ylabel(r"$x_i$")
plt.title(f"Loc-KBP-DLR covariance at T = {T}")
plt.show()
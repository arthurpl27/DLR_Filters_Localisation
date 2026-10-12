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
dt = 1e-3
N = int(T / dt)

c = 0.05
b = 0.05
r = 0.02

rank = 15
M = 30
k = 10

sigma = 1
gamma = 2

radius_index = 5
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

mathcal_L = L_neumann(d, h, c, r, b)

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


# ============================================================
#                    INITIAL DLR MEANS
# ============================================================

m_dlr = np.cos(2 * np.pi * x / L)
m_dlr_loc = m_dlr.copy()


# ============================================================
#              INITIAL DETERMINISTIC MODES
# ============================================================

U_raw = np.zeros((d, rank))

for j in range(1, rank + 1):

    U_raw[:, j - 1] = np.cos(2 * np.pi * j * x / L)


# ============================================================
#                INITIAL STOCHASTIC MODES
# ============================================================

Y_raw = rng.standard_normal((rank, M))

# ============================================================
#               INITIAL ORTHONORMALIZATION
# ============================================================

U, R_qr = np.linalg.qr(U_raw, mode="reduced")
Y = R_qr @ Y_raw

# Both filters start from exactly the same DLR approximation
U_loc = U.copy()
Y_loc = Y.copy()


# ============================================================
#                 INITIAL TRUE SIGNAL
# ============================================================

Y_true_0 = rng.standard_normal(rank)
X_true = m_dlr + U_raw @ Y_true_0



# ============================================================
#                    STORAGE ARRAYS
# ============================================================

true_signal = np.zeros((N + 1, d))
mean_signal_dlr = np.zeros((N + 1, d))
mean_signal_dlr_loc = np.zeros((N + 1, d))
rmse_values_dlr = np.zeros(N + 1)
rmse_values_dlr_loc = np.zeros(N + 1)

true_signal[0] = X_true
mean_signal_dlr[0] = m_dlr
mean_signal_dlr_loc[0] = m_dlr_loc
rmse_values_dlr[0] = np.sqrt(np.mean((m_dlr - X_true) ** 2))
rmse_values_dlr_loc[0] = np.sqrt(np.mean((m_dlr_loc - X_true) ** 2))

# ============================================================
#
#                  DLR-KBF vs Loc-DLR-KBF
#
# ============================================================

for n in range(N):

    # ========================================================
    #                     TRUE SIGNAL
    # ========================================================

    X_n = X_true.copy()


    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------

    dB_obs = np.sqrt(dt) * rng.standard_normal(k)
    dZ = H @ X_n * dt + sqrt_Gamma @ dB_obs

    # --------------------------------------------------------
    # True signal Euler-Maruyama update
    # --------------------------------------------------------

    dW_true = np.sqrt(dt) * rng.standard_normal(d)
    X_true = X_n + mathcal_L @ X_n * dt + sqrt_Sigma @ dW_true

    # ========================================================
    #                    PREDICTION STEP
    # ========================================================


    # --------------------------------------------------------
    # Mean prediction
    # --------------------------------------------------------

    m_hat = np.linalg.solve(I_d - dt * mathcal_L, m_dlr)
    m_hat_loc = np.linalg.solve(I_d - dt * mathcal_L, m_dlr_loc)

    # --------------------------------------------------------
    # Deterministic modes prediction: DLR
    # --------------------------------------------------------

    projector_perp = I_d - U @ U.T
    lhs_U = I_d - dt * projector_perp @ mathcal_L
    U_tilde = np.linalg.solve(lhs_U, U)

    # --------------------------------------------------------
    # Deterministic modes prediction: Loc-DLR
    # --------------------------------------------------------

    projector_perp_loc = I_d - U_loc @ U_loc.T
    lhs_U_loc = I_d - dt * projector_perp_loc @ mathcal_L
    U_tilde_loc = np.linalg.solve(lhs_U_loc, U_loc)

    # --------------------------------------------------------
    # SAME process noise for both filters
    # --------------------------------------------------------

    dW = np.sqrt(dt) * rng.standard_normal((d, M))

    # --------------------------------------------------------
    # Stochastic modes prediction: DLR
    # --------------------------------------------------------

    L_U = U.T @ mathcal_L @ U
    lhs_Y_prediction = I_rank - dt * L_U
    rhs_Y_prediction = Y + U.T @ sqrt_Sigma @ dW
    Y_tilde = np.linalg.solve(lhs_Y_prediction, rhs_Y_prediction)

    # --------------------------------------------------------
    # Stochastic modes prediction: Loc-DLR
    # --------------------------------------------------------

    L_U_loc = U_loc.T @ mathcal_L @ U_loc
    lhs_Y_prediction_loc = I_rank - dt * L_U_loc
    rhs_Y_prediction_loc = Y_loc + U_loc.T @ sqrt_Sigma @ dW
    Y_tilde_loc = np.linalg.solve(lhs_Y_prediction_loc, rhs_Y_prediction_loc)

    # ========================================================
    #                 QR RE-ORTHONORMALIZATION
    # ========================================================


    # --------------------------------------------------------
    # DLR
    # --------------------------------------------------------

    U_hat, R_qr = np.linalg.qr(U_tilde, mode="reduced")
    Y_hat = R_qr @ Y_tilde

    # --------------------------------------------------------
    # Loc-DLR
    # --------------------------------------------------------

    U_hat_loc, R_qr_loc = np.linalg.qr(U_tilde_loc, mode="reduced")
    Y_hat_loc = R_qr_loc @ Y_tilde_loc

    # ========================================================
    #               SAMPLE COVARIANCE OF Y
    # ========================================================

    C_Y_hat = (Y_hat @ Y_hat.T) / (M - 1)
    C_Y_hat_loc = (Y_hat_loc @ Y_hat_loc.T) / (M - 1)

    # ========================================================
    #                    ANALYSIS STEP
    # ========================================================


    # Common matrices
    mathcal_V = Gamma_inv
    mathcal_VH = mathcal_V @ H

    # ========================================================
    #                  DLR MEAN UPDATE
    # ========================================================

    matrix1 = U_hat @ Y_hat
    matrix2 = matrix1.T @ H.T
    # C_hat H^T
    mathcal_U = (matrix1 @ matrix2) / (M - 1)
    rhs_mean = m_hat + mathcal_U @ mathcal_V @ dZ
    woodbury_matrix = I_k + mathcal_VH @ mathcal_U * dt
    rhs_solve = mathcal_VH @ rhs_mean
    woodbury_solve = np.linalg.solve(woodbury_matrix, rhs_solve)
    m_new = rhs_mean - mathcal_U @ woodbury_solve * dt

    # ========================================================
    #                LOC-DLR MEAN UPDATE
    # ========================================================

    matrix1_loc = U_hat_loc @ Y_hat_loc
    matrix2_loc = matrix1_loc.T @ H.T
    # Raw C_hat H^T for the localized filter
    mathcal_U_loc_raw = (matrix1_loc @ matrix2_loc) / (M - 1)

    # --------------------------------------------------------
    # Localisation of C H^T
    #
    # Rows: all state locations x_i
    #
    # Columns: physical locations of observations
    #
    # Periodic domain size = L
    # --------------------------------------------------------

    mathcal_U_loc = covariance_loc_gc(radius_phys, mathcal_U_loc_raw, row_positions=x,
                                        col_positions=x[observation_indices], domain_size=L)

    rhs_mean_loc = m_hat_loc + mathcal_U_loc @ mathcal_V @ dZ
    woodbury_matrix_loc = I_k + mathcal_VH @ mathcal_U_loc * dt
    rhs_solve_loc = mathcal_VH @ rhs_mean_loc
    woodbury_solve_loc = np.linalg.solve(woodbury_matrix_loc, rhs_solve_loc)
    m_new_loc = rhs_mean_loc - mathcal_U_loc @ woodbury_solve_loc * dt

    # ========================================================
    #        SAME OBSERVATION NOISE FOR BOTH FILTERS
    # ========================================================

    dB = np.sqrt(dt) * rng.standard_normal((k, M))

    # ========================================================
    #              DLR STOCHASTIC MODES UPDATE
    # ========================================================

    Y_matrix = C_Y_hat @ U_hat.T @ H.T @ Gamma_inv @ H @ U_hat
    lhs_Y_analysis = I_rank + dt * Y_matrix
    rhs_Y_analysis = Y_hat - C_Y_hat @ U_hat.T @ H.T @ Gamma_inv_sqrt @ dB
    Y_new = np.linalg.solve(lhs_Y_analysis, rhs_Y_analysis)

    # ========================================================
    #           LOC-DLR STOCHASTIC MODES UPDATE
    # ========================================================

    lhs_Y_analysis_loc = I_rank + U_hat_loc.T @ mathcal_U_loc @ mathcal_VH @ U_hat_loc * dt
    rhs_Y_analysis_loc = Y_hat_loc - U_hat_loc.T @ mathcal_U_loc @ Gamma_inv_sqrt @ dB
    Y_new_loc = np.linalg.solve(lhs_Y_analysis_loc, rhs_Y_analysis_loc)

    # ========================================================
    #                    UPDATE VARIABLES
    # ========================================================

    # DLR
    m_dlr = m_new
    U = U_hat
    Y = Y_new

    # Loc-DLR
    m_dlr_loc = m_new_loc
    U_loc = U_hat_loc
    Y_loc = Y_new_loc


    # ========================================================
    #                       STORAGE
    # ========================================================

    true_signal[n + 1] = X_true
    mean_signal_dlr[n + 1] = m_dlr
    mean_signal_dlr_loc[n + 1] = m_dlr_loc
    rmse_values_dlr[n + 1] = np.sqrt(np.mean((m_dlr - X_true) ** 2))
    rmse_values_dlr_loc[n + 1] = np.sqrt(np.mean((m_dlr_loc - X_true) ** 2))

# ============================================================
#                 FINAL DLR APPROXIMATIONS
# ============================================================


# ------------------------------------------------------------
# DLR
# ------------------------------------------------------------

ensemble_dlr = m_dlr[:, None] + U @ Y
C_Y_final = (Y @ Y.T) / (M - 1)
C_dlr_final = U @ C_Y_final @ U.T

# ------------------------------------------------------------
# Loc-DLR
# ------------------------------------------------------------

ensemble_dlr_loc = m_dlr_loc[:, None] + U_loc @ Y_loc
C_Y_final_loc = (Y_loc @ Y_loc.T) / (M - 1)
C_dlr_final_loc = U_loc @ C_Y_final_loc @ U_loc.T

# ============================================================
#                     INFORMATION
# ============================================================

print("============================================")
print(f"State dimension d:          {d}")
print(f"DLR rank:                   {rank}")
print(f"Observation dimension k:    {k}")
print(f"Number of Y samples M:      {M}")
print(f"Localization radius:        {radius_phys}")
print("============================================")
print("DLR U shape:", U.shape)
print("DLR Y shape:", Y.shape)
print("Loc-DLR U shape:", U_loc.shape)
print("Loc-DLR Y shape:", Y_loc.shape)
print("Final DLR RMSE:", rmse_values_dlr[-1])
print("Final Loc-DLR RMSE:", rmse_values_dlr_loc[-1])
print("Final RMSE difference:", rmse_values_dlr_loc[-1] - rmse_values_dlr[-1])

# ============================================================
#              FILTERED MEANS — COMPARISON
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(x, true_signal[-1], label="True signal")
plt.plot(x, mean_signal_dlr[-1], linewidth=2, label="DLR-KBF mean")
plt.plot(x, mean_signal_dlr_loc[-1], linestyle="--", linewidth=2, marker="o", markevery=5,
         label="Loc-DLR-KBF mean")
plt.scatter(x[observation_indices], true_signal[-1, observation_indices], marker="x", label="Observed locations")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,T)$")
plt.title(f"DLR-KBF vs Loc-DLR-KBF at T = {T}, rank = {rank}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ============================================================
#                    RMSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(time, rmse_values_dlr, label="DLR-KBF")
plt.plot(time, rmse_values_dlr_loc, linestyle="--", label="Loc-DLR-KBF")
plt.xlabel(r"$t$")
plt.ylabel("RMSE")
plt.title("DLR-KBF vs Loc-DLR-KBF estimation error")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ============================================================
#                  A FEW DLR SAMPLES
# ============================================================

number_samples_plot = min(5, M)

fig, axes = plt.subplots(1,2, figsize=(14, 5))

# ------------------------------------------------------------
# DLR samples
# ------------------------------------------------------------

for p in range(number_samples_plot):

    axes[0].plot(x, ensemble_dlr[:, p], alpha=0.6, label=f"Sample {p + 1}")

axes[0].plot(x, m_dlr, linewidth=3, label="DLR mean")
axes[0].plot(x, X_true, linestyle="--", linewidth=2, label="True signal")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$u(x,T)$")
axes[0].set_title("DLR-KBF samples")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ------------------------------------------------------------
# Loc-DLR samples
# ------------------------------------------------------------

for p in range(number_samples_plot):

    axes[1].plot(x, ensemble_dlr_loc[:, p], alpha=0.6, label=f"Sample {p + 1}")

axes[1].plot(x, m_dlr_loc, linewidth=3, label="Loc-DLR mean")
axes[1].plot(x, X_true, linestyle="--", linewidth=2, label="True signal")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$u(x,T)$")
axes[1].set_title("Loc-DLR-KBF samples")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
#                  COVARIANCE COMPARISON
# ============================================================

# Same color scale for both covariance plots
vmin = min(C_dlr_final.min(), C_dlr_final_loc.min())
vmax = max(C_dlr_final.max(), C_dlr_final_loc.max())

fig, axes = plt.subplots(1,2, figsize=(14, 6))

# ------------------------------------------------------------
# DLR covariance
# ------------------------------------------------------------

im1 = axes[0].imshow(C_dlr_final, origin="lower", extent=[0, L, 0, L], aspect="auto", vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axes[0], label="Covariance")
axes[0].set_xlabel(r"$x_j$")
axes[0].set_ylabel(r"$x_i$")
axes[0].set_title("DLR-KBF covariance")

# ------------------------------------------------------------
# Loc-DLR covariance
# ------------------------------------------------------------

im2 = axes[1].imshow(C_dlr_final_loc, origin="lower", extent=[0, L, 0, L], aspect="auto", vmin=vmin, vmax=vmax)
fig.colorbar(im2, ax=axes[1], label="Covariance")
axes[1].set_xlabel(r"$x_j$")
axes[1].set_ylabel(r"$x_i$")
axes[1].set_title("Loc-DLR-KBF covariance")

plt.tight_layout()
plt.show()
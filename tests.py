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
T = 30
dt = 1e-3
N = int(T / dt)
seed = 42

c = 0.05
b = 0.05
r = 0.02

rank = 2
M = 30
k = d

#sigma = 1
gamma = 1

radius_index = 5
radius_phys = radius_index * h

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

H = np.eye(d)


def run_loc_comparison(sigma, gamma):

    rng = np.random.default_rng(seed)

    sqrt_Sigma = np.sqrt(sigma) * I_d
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

    # both filters start from exactly the same DLR approximation
    U_loc = U.copy()
    Y_loc = Y.copy()

    # ============================================================
    #                 INITIAL TRUE SIGNAL
    # ============================================================

    Y_true_0 = rng.standard_normal(rank)
    X_true = m_dlr + U_raw @ Y_true_0
    Z_n = np.zeros(k)

    # ============================================================
    #                    STORAGE ARRAYS
    # ============================================================

    true_signal = np.zeros((N + 1, d))
    mean_signal_dlr = np.zeros((N + 1, d))
    mean_signal_dlr_loc = np.zeros((N + 1, d))
    Z_process = np.zeros((N + 1, k))
    rmse_values_dlr = np.zeros(N + 1)
    rmse_values_dlr_loc = np.zeros(N + 1)
    mean_difference_norms = np.zeros(N+1)
    covariance_difference_norms = np.zeros(N+1)

    true_signal[0] = X_true
    mean_signal_dlr[0] = m_dlr
    mean_signal_dlr_loc[0] = m_dlr_loc
    observation_values = np.zeros((N + 1, k))
    rmse_values_dlr[0] = np.sqrt(np.mean((m_dlr - X_true) ** 2))
    rmse_values_dlr_loc[0] = np.sqrt(np.mean((m_dlr_loc - X_true) ** 2))
    mean_difference_norms[0] = 0          # both filters have same initial condition
    covariance_difference_norms[0] = 0    # both filters have same initial condition


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
        Z_n += dZ
        Z_process[n + 1] = Z_n
        observation_values[n + 1] = dZ / dt

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
        # Model noise for both filters
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

        # ========================================================
        #                    ANALYSIS STEP
        # ========================================================

        mathcal_V = Gamma_inv
        mathcal_VH = mathcal_V @ H

        # --------------------------------------------------------
        #                  DLR MEAN UPDATE
        # --------------------------------------------------------
        matrix1 = U_hat @ Y_hat
        matrix2 = matrix1.T @ H.T
        # C_hat H^T
        mathcal_U = (matrix1 @ matrix2) / (M - 1)
        rhs_mean = m_hat + mathcal_U @ mathcal_V @ dZ
        woodbury_matrix = I_k + mathcal_VH @ mathcal_U * dt
        rhs_solve = mathcal_VH @ rhs_mean
        woodbury_solve = np.linalg.solve(woodbury_matrix, rhs_solve)
        m_new = rhs_mean - mathcal_U @ woodbury_solve * dt

        # --------------------------------------------------------
        #                LOC-DLR MEAN UPDATE
        # --------------------------------------------------------

        matrix1_loc = U_hat_loc @ Y_hat_loc
        matrix2_loc = matrix1_loc.T @ H.T
        # Raw C_hat H^T for the localized filter
        mathcal_U_loc_raw = (matrix1_loc @ matrix2_loc) / (M - 1)
        mathcal_U_loc = covariance_loc_gc(radius_phys, mathcal_U_loc_raw, row_positions=x, col_positions=x)
        rhs_mean_loc = m_hat_loc + mathcal_U_loc @ mathcal_V @ dZ
        woodbury_matrix_loc = I_k + mathcal_VH @ mathcal_U_loc * dt
        rhs_solve_loc = mathcal_VH @ rhs_mean_loc
        woodbury_solve_loc = np.linalg.solve(woodbury_matrix_loc, rhs_solve_loc)
        m_new_loc = rhs_mean_loc - mathcal_U_loc @ woodbury_solve_loc * dt

        # --------------------------------------------------------
        #                  OBSERVATION NOISE
        # --------------------------------------------------------

        dB = np.sqrt(dt) * rng.standard_normal((k, M))

        # --------------------------------------------------------
        #              DLR STOCHASTIC MODES UPDATE
        # --------------------------------------------------------

        Y_matrix = C_Y_hat @ U_hat.T @ H.T @ Gamma_inv @ H @ U_hat
        lhs_Y_analysis = I_rank + dt * Y_matrix
        rhs_Y_analysis = Y_hat - C_Y_hat @ U_hat.T @ H.T @ Gamma_inv_sqrt @ dB
        Y_new = np.linalg.solve(lhs_Y_analysis, rhs_Y_analysis)

        # --------------------------------------------------------
        #           LOC-DLR STOCHASTIC MODES UPDATE
        # --------------------------------------------------------

        lhs_Y_analysis_loc = I_rank + U_hat_loc.T @ mathcal_U_loc @ mathcal_VH @ U_hat_loc * dt
        rhs_Y_analysis_loc = Y_hat_loc - U_hat_loc.T @ mathcal_U_loc @ Gamma_inv_sqrt @ dB
        Y_new_loc = np.linalg.solve(lhs_Y_analysis_loc, rhs_Y_analysis_loc)

        # --------------------------------------------------------
        #                    UPDATE VARIABLES
        # --------------------------------------------------------

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

        # Mean differences
        mean_difference_norms[n + 1] = np.linalg.norm(m_dlr - m_dlr_loc, ord=2)

        # Covariances differences
        C_dlr = (Y @ Y.T) / (M-1)
        C_dlr_full = U @ C_dlr @ U.T
        C_loc_dlr = (Y_loc @ Y_loc.T) / (M-1)
        C_loc_dlr_full = U_loc @ C_loc_dlr @ U_loc.T
        covariance_difference_norms[n + 1] = np.linalg.norm(C_dlr_full - C_loc_dlr_full, ord="fro")

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

    return {
        # Truth
        "true_signal": true_signal,

        # Observation process
        "Z_process": Z_process,

        # Delta Z / Delta t
        "observation_values": observation_values,

        # Mean histories
        "mean_signal_dlr": mean_signal_dlr,
        "mean_signal_dlr_loc": mean_signal_dlr_loc,

        # RMSE histories
        "rmse_values_dlr": rmse_values_dlr,
        "rmse_values_dlr_loc": rmse_values_dlr_loc,

        # Final means
        "m_dlr": m_dlr,
        "m_dlr_loc": m_dlr_loc,

        # Final deterministic modes
        "U": U,
        "U_loc": U_loc,

        # Final stochastic modes
        "Y": Y,
        "Y_loc": Y_loc,

        # Final ensembles
        "ensemble_dlr": ensemble_dlr,
        "ensemble_dlr_loc": ensemble_dlr_loc,

        # Reduced covariances
        "C_Y_final": C_Y_final,
        "C_Y_final_loc": C_Y_final_loc,

        # Full covariances
        "C_dlr_final": C_dlr_final,
        "C_dlr_final_loc": C_dlr_final_loc,

        # Mean and covariance differences
        "mean_difference_norms": mean_difference_norms,
        "covariance_difference_norms": covariance_difference_norms}


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

sigma_values = [0.001, 0.01, 0.1, 0.5, 1, 2]
#gamma_values = [0.001, 0.01, 0.1, 0.5, 1, 2]
results = {}

for sigma in sigma_values:

    print(f"Running sigma = {sigma}")

    results[sigma] = run_loc_comparison(sigma,gamma)

# ============================================================
#              FILTERED MEANS — COMPARISON
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10),sharex=True)
axes = axes.ravel()

for ax, sigma in zip(axes, sigma_values):

    result = results[sigma]
    true_signal = result["true_signal"]
    mean_signal_dlr = result["mean_signal_dlr"]
    mean_signal_dlr_loc = result["mean_signal_dlr_loc"]

    # ------------------------------------------------------------
    #                   STATE / FILTER MEANS
    # ------------------------------------------------------------

    ax.plot(x, true_signal[-1], label="True signal")
    ax.plot(x, mean_signal_dlr[-1], linewidth=2, label="DLR-KBF")
    ax.plot(x, mean_signal_dlr_loc[-1], linestyle="--", linewidth=2, label="Loc-DLR-KBF")
    ax.set_xlabel(r"$x$")
    ax.grid(alpha=0.3)


    # ------------------------------------------------------------
    #                    OBSERVATION PROCESS Z_T
    # ------------------------------------------------------------
    ax_obs = ax.twinx()
    ax_obs.scatter(x, result["Z_process"][-1], marker="x", s=15, alpha=0.6, label=r"$Z_T$")
    ax_obs.set_ylabel(r"Observation $Z_T$")
    ax.set_title(rf"$\sigma={sigma}$")

# ------------------------------------------------------------
#                           LEGEND
# ------------------------------------------------------------

lines1, labels1 = axes[0].get_legend_handles_labels()
ax_obs_0 = axes[0].twinx()
fig.suptitle(rf"DLR-KBF vs Loc-DLR-KBF at T={T}, "rf"rank={rank}, $\gamma={gamma}$, "rf"radius={radius_index}, M={M}",
    fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================
#                    RMSE COMPARISON
# ============================================================

fig, axes = plt.subplots(2,3, figsize=(18, 10), sharex=True)

axes = axes.ravel()
for ax, sigma in zip(axes, sigma_values):

    result = results[sigma]
    ax.plot(time, result["rmse_values_dlr"], label="DLR-KBF")
    ax.plot(time, result["rmse_values_dlr_loc"], linestyle="--", label="Loc-DLR-KBF")
    ax.set_title(rf"$\sigma={sigma}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3)

axes[0].legend()
fig.suptitle(rf"DLR-KBF vs Loc-DLR-KBF estimation error for $\gamma$ = {gamma}, radius = {radius_index}, M = {M}, rank = {rank}", fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================
#                  A FEW DLR SAMPLES
# ============================================================

number_samples_plot = min(5, M)

fig, axes = plt.subplots(2, len(sigma_values), figsize=(24, 8), sharex=True, sharey=True)

for j, sigma in enumerate(sigma_values):

    result = results[sigma]
    ensemble_dlr = result["ensemble_dlr"]
    ensemble_dlr_loc = result["ensemble_dlr_loc"]
    m_dlr = result["m_dlr"]
    m_dlr_loc = result["m_dlr_loc"]
    X_true = result["true_signal"][-1]

    # ========================================================
    # DLR-KBF
    # ========================================================

    for p in range(number_samples_plot):

        axes[0, j].plot(x, ensemble_dlr[:, p], alpha=0.5)

    axes[0, j].plot(x, m_dlr, linewidth=3, label="Mean")
    axes[0, j].plot(x, X_true, linestyle="--", linewidth=2, label="True signal")
    axes[0, j].set_title(rf"DLR, $\sigma={sigma}$")
    axes[0, j].grid(alpha=0.3)

    # ========================================================
    # Loc-DLR-KBF
    # ========================================================

    for p in range(number_samples_plot):

        axes[1, j].plot(x, ensemble_dlr_loc[:, p], alpha=0.5)

    axes[1, j].plot(x, m_dlr_loc, linewidth=3, label="Mean")
    axes[1, j].plot(x, X_true, linestyle="--", linewidth=2, label="True signal")
    axes[1, j].set_title(rf"Loc-DLR, $\sigma={sigma}$")
    axes[1, j].set_xlabel(r"$x$")
    axes[1, j].grid(alpha=0.3)

axes[0, 0].set_ylabel(r"$u(x,T)$")
axes[1, 0].set_ylabel(r"$u(x,T)$")
axes[0, 0].legend()
fig.suptitle(rf"DLR ensembles at T = {T}, rank = {rank}, $\gamma$ = {gamma}, radius = {radius_index}, M = {M}", fontsize=16)
plt.tight_layout()
plt.show()


# ============================================================
#              MEAN DIFFERENCE NORM
# ============================================================

plt.figure(figsize=(10, 6))

for sigma in sigma_values:

    plt.plot(time, results[sigma]["mean_difference_norms"], label=rf"$\sigma={sigma}$")

plt.xlabel(r"$t$")
plt.ylabel(
    r"$\|m_t^{\mathrm{DLR-KBP}}"
    r"-m_t^{\mathrm{Loc-DLR-KBP}}\|_2$")
plt.title(rf"2-norm between DLR-KBP and Loc-DLR-KBP means, rank = {rank}, $\gamma$ = {gamma}, radius = {radius_index}, M = {M}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
#              COVARIANCE DIFFERENCE NORM
# ============================================================

plt.figure(figsize=(10, 6))

for sigma in sigma_values:

    plt.plot(time, results[sigma]["covariance_difference_norms"], label=rf"$\sigma={sigma}$")

plt.xlabel(r"$t$")
plt.ylabel(
    r"$\|C_t^{\mathrm{DLR-KBP}}"
    r"-C_t^{\mathrm{Loc-DLR-KBP}}\|_F$")
plt.title(rf"Frobenius norm between DLR-KBP and Loc-DLR-KBP covariances, rank = {rank}, $\gamma$ = {gamma}, radius = {radius_index}, M = {M}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

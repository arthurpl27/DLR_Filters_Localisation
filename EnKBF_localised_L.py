import numpy as np
import matplotlib.pyplot as plt
from Localisation_tools import covariance_loc_gc
from PDE_tools import L_periodic, L_neumann

# ============================================================
#                    PARAMETERS & SEED
# ============================================================

L = 10
d = 100
h = L / d
T = 1
dt = 1e-4
N = int(T / dt)
c = 1
r = 0.1
b = 0.2
sigma = 0.1           # 0.001 or 0.01 or 0.1        # particles
beta = 0.1              # 0.2                        # observations
M = 20                # number of particles
k = 10                # number of observations
radius_index = 5            # localization radius
radius_phys = radius_index * h

seed = 42
rng = np.random.default_rng(seed)

# ============================================================
#         SPATIAL GRID & INITIAL CONDITION & OPERATOR
# ============================================================

x = np.arange(d) * h                     # spatial grid
time = np.linspace(0, T, N + 1)   # time grid

U0 = np.sin(2 * np.pi * x / L)                                      # initial condition
#U0 = np.sin(2 * np.pi * x / L) + rng.standard_normal(d)           # random initial condition

mathcal_L = L_periodic(d,h,c,r,b)   # discretized operator

# ============================================================
#               SIGNAL & OBSERVATION MATRICES
# ============================================================

Q = sigma * np.eye(d)
sqrt_Q = np.sqrt(sigma) * np.eye(d)

observation_indices = np.linspace(0, d - 1, k, dtype=int)    # observe k equally spaced components
H = np.zeros((k, d))

for i, j in enumerate(observation_indices):
    H[i, j] = 1

R = beta * np.eye(k)
R_inv = (1/beta) * np.eye(k)
sqrt_R = np.sqrt(beta) * np.eye(k)

I_d = np.eye(d)  # Identity matrix of dimension d
I_k = np.eye(k)  # Identity matrix of dimension k


# ============================================================
#                   INITIALISATIONS
# ============================================================

X_true = U0 + rng.standard_normal(d)   # initial true signal

ensemble_0 = U0[:, None] + rng.standard_normal((d, M))   # initial ensemble
                                                           # convention is ensemble[:, p] = particle p


ensemble_hat = ensemble_0.copy()
ensemble_loc = ensemble_0.copy()
mean_hat_0 = ensemble_0.mean(axis=1)        # initial mean

true_signal = np.zeros((N + 1, d))
mean_signal_hat = np.zeros((N + 1, d))
mean_signal_loc = np.zeros((N + 1, d))

true_signal[0] = X_true
mean_signal_hat[0] = mean_hat_0
mean_signal_loc[0] = mean_signal_hat[0]

# RMSE
rmse_values_hat = np.zeros(N + 1)
rmse_values_loc = np.zeros(N + 1)
rmse_values_hat[0] = np.sqrt(np.mean((mean_hat_0 - X_true)**2))
rmse_values_loc[0] = rmse_values_hat[0]

# =============================================================================
#                                  EnKBF
#           Euler-Maruyama for discretisation + semi-implicit scheme
# =============================================================================

for n in range(N):

    X_n = X_true.copy()    # current state

    dB_obs = np.sqrt(dt) * rng.standard_normal(k)
    dZ = H @ X_n * dt + sqrt_R @ dB_obs    # observation

    dW_true = np.sqrt(dt) * rng.standard_normal(d)
    X_true = X_n + mathcal_L @ X_n * dt + sqrt_Q @ dW_true    # true signal

    # --------------------------------------------------------
    #                   UPDATE ENSEMBLE
    #                    Coupled SDEs
    # --------------------------------------------------------

    dW_particles = np.sqrt(dt) * rng.standard_normal((d, M))
    dB_particles = np.sqrt(dt) * rng.standard_normal((k, M))

    noisy_observation = dZ[:, None] + sqrt_R @ dB_particles  # noisy observations: ΔZ_n + R^{1/2} Δ(B^tilde)_n^(p)


    # prediction step
    prediction_hat = ensemble_hat + mathcal_L @ ensemble_hat * dt + sqrt_Q @ dW_particles
    prediction_loc = ensemble_loc + mathcal_L @ ensemble_loc * dt + sqrt_Q @ dW_particles

    # ======  NON-LOCALISED COVARIANCE  ============= #

    prediction_mean_hat = prediction_hat.mean(axis=1)
    prediction_star_hat = prediction_hat - prediction_mean_hat[:, None]
    intermediate_step = prediction_star_hat.T @ H.T
    K_hat = (prediction_star_hat @ intermediate_step @ R_inv) / (M - 1)

    lhs_hat = I_k + H @ K_hat * dt
    rhs_hat = noisy_observation - H @ prediction_hat * dt
    solve_hat = np.linalg.solve(lhs_hat, rhs_hat)
    new_ensemble_hat = prediction_hat + K_hat @ solve_hat

    # ===========  LOCALISED COVARIANCE  ============= #

    prediction_mean_loc = prediction_loc.mean(axis=1)
    prediction_star_loc = prediction_loc - prediction_mean_loc[:, None]
    C_loc_raw = (prediction_star_loc @ prediction_star_loc.T) / (M - 1)
    C_loc = covariance_loc_gc(radius_phys, C_loc_raw, row_positions=x)     # covariance localisation
    K_loc = C_loc @ H.T @ R_inv

    lhs_loc = I_k + H @ K_loc * dt
    rhs_loc = noisy_observation - H @ prediction_loc * dt
    solve_loc = np.linalg.solve(lhs_loc, rhs_loc)
    new_ensemble_loc = prediction_loc + K_loc @ solve_loc

    # ===========  UPDATE ENSEMBLE  ============= #

    mean_hat = new_ensemble_hat.mean(axis=1)  # mean of analysis step
    mean_loc = new_ensemble_loc.mean(axis=1)
    ensemble_hat = new_ensemble_hat  # update ensemble
    ensemble_loc = new_ensemble_loc
    true_signal[n + 1] = X_true
    mean_signal_hat[n + 1] = mean_hat
    mean_signal_loc[n + 1] = mean_loc
    rmse_values_hat[n + 1] = np.sqrt(np.mean((mean_hat - X_true)**2))
    rmse_values_loc[n + 1] = np.sqrt(np.mean((mean_loc - X_true) ** 2))


print("============================================")
print(f"State dimension d:            {d}")
print(f"Observation dimension k:      {k}")
print(f"Ensemble size M:              {M}")
print("============================================")

# ============================================================
#                         PLOTS
# ============================================================

# ------------------------------------------------------------
#           True state vs EnKBF estimate at final time
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(x, U0, linestyle="--", label=r"$U_0(x)$")
plt.plot(x, true_signal[-1], label="True signal at T")
plt.plot(x, mean_signal_hat[-1], label="EnKBF mean hat")
plt.plot(x, mean_signal_loc[-1], label="EnKBF mean loc")
# Observed locations
plt.scatter(x[observation_indices], true_signal[-1, observation_indices], marker="x", label="Observed locations")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,T)$")
plt.title(f"EnKBF with d = {d} and k = {k} at T = {T}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ------------------------------------------------------------
#                      RMSE in time
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(time, rmse_values_hat, label="EnKBF RMSE non-loc")
plt.plot(time, rmse_values_loc, label="EnKBF RMSE loc")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel("RMSE")
plt.title("EnKBF Estimation Error")
plt.grid(alpha=0.3)
plt.show()


# ------------------------------------------------------------
#              A FEW PARTICLES AT FINAL TIME
# ------------------------------------------------------------

number_particles_plot = 5

plt.figure(figsize=(10, 5))

for p in range(number_particles_plot):
    plt.plot(x, ensemble_hat[:, p], alpha=0.7, label=f"Particle {p + 1}")

plt.plot(x, mean_signal_hat[-1], linewidth=3, label="EnKBF mean")
plt.plot(x, true_signal[-1], linewidth=3, linestyle="--", label="True signal")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,T)$")
plt.title("Non-localised EnKBF particles at final time")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ============================================================
#                RAW VS LOCALISED COVARIANCE
# ============================================================

mean_loc_final = ensemble_loc.mean(axis=1)                                         # final mean
ensemble_star_loc_final = ensemble_loc - mean_loc_final[:, None]
C_loc_raw_final = (ensemble_star_loc_final @ ensemble_star_loc_final.T) / (M - 1)  # final empirical covariance
C_loc_final = covariance_loc_gc(radius_phys, C_loc_raw_final, row_positions=x)       # final localized covariance

# Same color scale for both plots
vmin = min(C_loc_raw_final.min(), C_loc_final.min())
vmax = max(C_loc_raw_final.max(), C_loc_final.max())

fig, axes = plt.subplots(1,2, figsize=(14, 6))

im1 = axes[0].imshow(C_loc_raw_final, origin="lower", extent=[0, L, 0, L], aspect="auto", vmin=vmin, vmax=vmax)
fig.colorbar(im1, ax=axes[0], label="Covariance")
axes[0].set_xlabel(r"$x_j$")
axes[0].set_ylabel(r"$x_i$")
axes[0].set_title("Raw empirical covariance")

im2 = axes[1].imshow(C_loc_final, origin="lower", extent=[0, L, 0, L], aspect="auto", vmin=vmin, vmax=vmax)
fig.colorbar(im2, ax=axes[1], label="Covariance")
axes[1].set_xlabel(r"$x_j$")
axes[1].set_ylabel(r"$x_i$")
axes[1].set_title(f"Gaspari-Cohn localised covariance, radius = {radius_index}")
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PDE_tools import L_periodic, L_neumann


# ============================================================
#                    PARAMETERS & SEED
# ============================================================

L = 10
d = 100
h = L / d
T = 1
dt = 1e-3
N = int(T / dt)
c = 1
r = 0.1
b = 0.2
sigma = 0.1
M = int(1e2)          # number of particles
k = 20                # number of observations

seed = 42
rng = np.random.default_rng(seed)

# ============================================================
#         SPATIAL GRID & INITIAL CONDITION & OPERATOR
# ============================================================

x = np.arange(d) * h                     # spatial grid
time = np.linspace(0, T, N + 1)   # time grid

U0 = np.sin(2 * np.pi * x / L)            # initial condition

mathcal_L = L_neumann(d,h,c,r,b)   # discretized operator

# ============================================================
#               SIGNAL & OBSERVATION MATRICES
# ============================================================

Q = sigma * np.eye(d)
sqrt_Q = np.sqrt(sigma) * np.eye(d)

observation_indices = np.linspace(0, d - 1, k, dtype=int)    # observe k equally spaced components

H = np.zeros((k, d))

for i, j in enumerate(observation_indices):
    H[i, j] = 1


R = 2 * np.eye(k)
R_inv = 0.5 * np.eye(k)
sqrt_R = np.sqrt(2) * np.eye(k)

I_d = np.eye(d)  # Identity matrix of dimension d
I_k = np.eye(k)  # Identity matrix of dimension k


# ============================================================
#                   INITIALISATIONS
# ============================================================

X_true = U0 + rng.standard_normal(d)   # initial true signal

ensemble_0 = U0[:, None] + rng.standard_normal((d, M))   # initial ensemble
                                                            # convention is ensemble[:, p] = particle p


ensemble = ensemble_0.copy()
mean_0 = ensemble_0.mean(axis=1)        # initial mean

true_signal = np.zeros((N + 1, d))
mean_signal = np.zeros((N + 1, d))

true_signal[0] = X_true
mean_signal[0] = mean_0

# RMSE
rmse_values = np.zeros(N + 1)
rmse_values[0] = np.sqrt(np.mean((mean_0 - X_true)**2))


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

    # ========================================================
    #                    WOODBURY METHOD
    # ========================================================

    # prediction step: X_n^{(p)} + L X_n^{(p)} Δt + Q^{1/2} ΔW_n^{(p)}
    prediction = ensemble + mathcal_L @ ensemble * dt + sqrt_Q @ dW_particles

    prediction_mean = prediction.mean(axis=1)   # mean of prediction step
    prediction_star = prediction - prediction_mean[:, None]  # centred mean of prediction step

    intermediate_step = prediction_star.T @ H.T
    K = (prediction_star @ intermediate_step @ R_inv) / (M - 1)  # construct K = C_hat H^T R^{-1}

    ## semi-implicit scheme ##
    lhs = I_k + H @ K * dt  # dim(HK) = k x k
    rhs = noisy_observation - H @ prediction * dt
    solve = np.linalg.solve(lhs, rhs)     # inverse
    new_ensemble= prediction + K @ solve   # new ensemble


    mean = new_ensemble.mean(axis=1)  # mean of analysis step
    #ensemble_star = (new_ensemble - mean[:, None])    # centred mean of analysis step
    #C = (ensemble_star @ ensemble_star.T) / (M - 1)  # covariance of analysis step

    ensemble = new_ensemble  # update ensemble
    true_signal[n + 1] = X_true
    mean_signal[n + 1] = mean
    rmse_values[n + 1] = np.sqrt(np.mean((mean - X_true)**2))


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
plt.plot(x, mean_signal[-1], label="EnKBF mean")
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
plt.plot(time, rmse_values)
plt.xlabel(r"$t$")
plt.ylabel("RMSE")
plt.title("EnKBF Estimation Error")
plt.grid(alpha=0.3)
plt.show()


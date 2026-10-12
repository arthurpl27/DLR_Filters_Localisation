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
sigma = 0.001           # 0.001 or 0.01 or 0.1        # particles
beta = 0.1              # 0.2                        # observations
M = 20                # number of particles
k = 10                # number of observations
radius = 10            # localization radius

seed = 42
rng = np.random.default_rng(seed)

# ============================================================
#         SPATIAL GRID & INITIAL CONDITION & OPERATOR
# ============================================================

x = np.arange(d) * h                     # spatial grid
time = np.linspace(0, T, N + 1)   # time grid

U0 = np.sin(2 * np.pi * x / L)            # initial condition

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

    # ===========  LOCALISE C  ============= #

    prediction_mean_loc = prediction_loc.mean(axis=1)
    prediction_star_loc = prediction_loc - prediction_mean_loc[:, None]
    C_loc_raw = (prediction_star_loc @ prediction_star_loc.T) / (M - 1)
    C_loc = covariance_loc_gc(radius, C_loc_raw, row_positions=x)     # covariance localisation
    K_loc = C_loc @ H.T @ R_inv

    lhs_loc = I_k + H @ K_loc * dt
    rhs_loc = noisy_observation - H @ prediction_loc * dt
    solve_loc = np.linalg.solve(lhs_loc, rhs_loc)
    new_ensemble_loc = prediction_loc + K_loc @ solve_loc

    # ======  LOCALISE CH.T and HCH.T  ============= #

    prediction_mean_hat = prediction_hat.mean(axis=1)
    prediction_star_hat = prediction_hat - prediction_mean_hat[:, None]
    intermediate_step = prediction_star_hat.T @ H.T
    CHT = prediction_star_hat @ intermediate_step / (M - 1)
    HCHT = intermediate_step.T @ intermediate_step / (M - 1)
    CHT_loc = covariance_loc_gc(radius, CHT, row_positions=np.arange(d), col_positions=observation_indices, domain_size=d)
    HCHT_loc = covariance_loc_gc(radius, HCHT, row_positions=observation_indices, col_positions=observation_indices,
                                                                                                        domain_size=d)
    K_hat_loc1 = CHT_loc
    K_hat_loc2 = HCHT_loc

    lhs_hat = R + K_hat_loc2 * dt
    rhs_hat = noisy_observation - H @ prediction_hat * dt
    solve_hat = np.linalg.solve(lhs_hat, rhs_hat)
    new_ensemble_hat = prediction_hat + K_hat_loc1 @ solve_hat


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
plt.plot(x, mean_signal_hat[-1], linestyle="-", linewidth=2, label=r"Localisation of $CH^\top$ and $HCH^\top$")
plt.plot(x, mean_signal_loc[-1], linestyle="--", marker="o", markevery=5, linewidth=2, label=r"Localisation of $C$")
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
plt.plot(time, np.abs(rmse_values_hat-rmse_values_loc), label="EnKBF Difference in RMSE", linestyle="--")
plt.legend()
plt.xlabel(r"$t$")
plt.xlabel(r"$t$")
plt.ylabel("RMSE")
plt.grid(alpha=0.3)
plt.show()


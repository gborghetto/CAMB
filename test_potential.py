import camb
import numpy as np
import matplotlib.pyplot as plt
from camb import dark_energy

# ── parameters ──────────────────────────────────────────────
FUNC_FILE  = '/Users/giuliaborghetto/CAMB_original/function_library/B_maths/compl_4/unique_equations_4.txt'
FUNC_INDEX = 120       # pow(Abs(a0),(x/2))  [1-based]
A0_VALUE   = 0.1
N_VALUE    = 2.0
THETA_I    = 0.0
V0         = 1e-8

H0    = 67
ombh2 = 0.02246
omch2 = 0.11896
tau   = 0.0559
ns    = 0.9686
As    = 1e-10 * np.exp(3.045)

z_obs = np.linspace(0, 2.4, 500)
a_obs = 1 / (1 + z_obs)

# ── CAMB run ─────────────────────────────────────────────────
de = dark_energy.QuintessenceInterp()
de.set_params(
    esr_param_a0=A0_VALUE,
    esr_functions_file=FUNC_FILE,
    esr_potential_index=FUNC_INDEX - 1,
    a_min=1e-4, a_max=1.0, n_a=500,
    n=N_VALUE, V0=V0, theta_i=THETA_I
)

pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, ns=ns, As=As)
pars.DarkEnergy = de
results = camb.get_results(pars)

# ── background quantities ─────────────────────────────────────
wde      = np.array(results.get_dark_energy_rho_w(a_obs)).T
phi, _   = results.get_dark_energy_phi_phidot(a_obs)

# ── V(a, phi) along background trajectory ────────────────────
V_camb   = results.get_dark_energy_Vphi(a_obs, phi, deriv=0)

# ── analytic V(a, phi) ───────────────────────────────────────
V0_tuned = results.Params.DarkEnergy.V0
B_analytic = np.exp((A0_VALUE - a_obs)**2)
V_analytic = V0_tuned * np.exp(-N_VALUE * phi) * B_analytic

# ── plot ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

# w(z)
ax1.plot(z_obs, wde[:, 1], color='orange', linewidth=2, label='CAMB')
ax1.axhline(-1, color='black', linestyle='-.', linewidth=1)
ax1.set_xlabel(r'$z$', fontsize=12)
ax1.set_ylabel(r'$w_{\mathrm{DE}}$', fontsize=12)
ax1.set_xlim(0, 2.5)
ax1.set_ylim([-1.5, -0.5])
ax1.legend()

# phi(z)
ax2.plot(z_obs, phi, color='orange', linewidth=2)
ax2.set_xlabel(r'$z$', fontsize=12)
ax2.set_ylabel(r'$\phi(z)$', fontsize=12)
ax2.set_xlim(0, 2.5)

# V(a, phi) — CAMB vs analytic
ax3.plot(phi, V_camb,     color='orange', linewidth=2, label='CAMB')
ax3.plot(phi, V_analytic, 'k--',          linewidth=1.5, label='Analytic')
ax3.set_xlabel(r'$\phi$', fontsize=12)
ax3.set_ylabel(r'$V(a,\phi)$', fontsize=12)
ax3.legend()

plt.tight_layout()
#plt.savefig('V_aphi_test.png', dpi=150)
plt.show()

print(f"Max relative error in V: {np.max(np.abs((V_camb - V_analytic) / (np.abs(V_analytic) + 1e-30))):.2e}")

B_train = np.array(list(de.B_train))
a_train = np.array(list(de.a_train))


plt.figure()
plt.plot(a_train, B_train, label='B_train')
plt.plot(a_train, np.exp((A0_VALUE - a_train)**2), '--', label='Analytic B(a)')
plt.xlabel('a'); plt.ylabel('B(a)')
plt.legend()
#plt.savefig('B_check.png', dpi=150)
plt.show()
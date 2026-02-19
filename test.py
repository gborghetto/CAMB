import camb
import numpy as np
import matplotlib.pyplot as plt
z1 = np.logspace(7,-2,500)
z2 = np.linspace(0.01,0.)
zs = np.concatenate((z1,z2))
z_obs = np.linspace(0,2.4,500)

# set up your params as usual
theta_i = 0.
omch2 = 1.172874274e-01 #0.12043 #1.215000088e-01
H0 = 6.706942438e+01 #67.77 #6.917809245e+01
ombh2 = 2.257427266e-02 #0.02262 #2.286772303e-02
As = 2.121895487e-09 #1e-10*np.exp(3.085) #2.153551844e-09
ns =  9.714166364e-01 #0.9640 #0.97
tau = 6.189654335e-02 #0.0769 #0.0571 #6.234300691e-02
pars = camb.set_params( ombh2=ombh2, omch2=omch2, omk=0.0, As = As, ns = ns, tau = tau, H0=H0, n =2,
                       dark_energy_model='QuintessenceModel',model_idx = 2, theta_i=theta_i, V0 = 1.0e-07,
                       )
camb.model.AccuracyParams(AccuracyBoost=5.,BackgroundTimeStepBoost=5.)
results = camb.get_results(pars);
# get phi and a arrays from the background evolution
ev_phi = np.array(results.get_dark_energy_phi_phidot(1/(1+z_obs))).T
a = 1/(1+z_obs)
# evaluate V at those (a, phi) pairs
V = results.get_dark_energy_Vphi(a, ev_phi[:,0], deriv=0)
dV = results.get_dark_energy_Vphi(a, ev_phi[:,0], deriv=1)

# plt.plot(ev_phi[:,0],V)

wde = np.array(results.get_dark_energy_rho_w(1/(1+z_obs))).T
#plt.plot(z_obs,wde[:,1])
plt.plot(z_obs,V)

plt.show()
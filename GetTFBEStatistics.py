"""
Stand alone script to calculate the Wigner functions of
Thomas-Fermi and Bose-Einstein statistics
"""
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# Specify Hamiltonian of the system
V = lambda x: -0.05*x**2 + 0.03*x**4
K = lambda p: 0.5*p**2

# Chemical potential
mu = 0

# Inverse temperature for final state
beta = 1.5

####################################################
#
# Parameters of for both numerical algorithms
# (to find Gibbs state and to propagate Moyal equation)
#
####################################################

X_gridDIM = 512       # Discretization grid size in X
P_gridDIM = 512         # Discretization grid size in P

X_amplitude = 5.         # Window range -X_amplitude to X_amplitude
P_amplitude = 5.        # Window range -P_amplitude to P_amplitude

# Discretization resolution
dX = 2.*X_amplitude/float(X_gridDIM)
dP = 2.*P_amplitude/float(P_gridDIM)

dTheta = 2.*np.pi/(2.*P_amplitude)
Theta_amplitude = dTheta*P_gridDIM/2.
dLambda = 2.*np.pi/(2.*X_amplitude)
Lambda_amplitude = dLambda*X_gridDIM/2.

# vectors with range of coordinates
X_range = np.linspace(-X_amplitude, X_amplitude - dX , X_gridDIM )
Lambda_range = np.linspace(-Lambda_amplitude, Lambda_amplitude-dLambda, X_gridDIM)

Theta_range = np.linspace(-Theta_amplitude, Theta_amplitude - dTheta, P_gridDIM)
P_range = np.linspace(-P_amplitude, P_amplitude - dP, P_gridDIM)

# matrices of grid of coordinates
X = fftpack.fftshift(X_range)[np.newaxis,:]
Theta = fftpack.fftshift(Theta_range)[:,np.newaxis]

Lambda = fftpack.fftshift(Lambda_range)[np.newaxis,:]
P = fftpack.fftshift(P_range)[:,np.newaxis]

#################################################################################
#
# Get Gibbs state
#
#################################################################################

# Calculate dbeta depending on the specified value of beta
# In particular, dbeta cannot exceed beta and the fraction beta/dbeta must be integer
max_dbeta = 0.01 # max 1/(kT) increment
BetaIterSteps = int(np.ceil(beta/min(max_dbeta, beta)))
dbeta = beta/BetaIterSteps

# Pre-calculate exp taking into account the chemical potential

VV = V(X - 0.5*Theta) + V(X + 0.5*Theta)
VV -= VV.min()

expV = np.exp(-0.5*dbeta*(VV - mu))
expK = np.exp(-0.5*dbeta*(K(P + 0.5*Lambda) + K(P - 0.5*Lambda)))

Hamiltonian = K(P) + V(X)
previous_energy = np.inf

# Initialize Wigner function to be used in beta propagation
W = np.ones((P.size, X.size), dtype=np.complex)

# Even and odd power in taylor expansion in the exact Wigner function
W_even_pows = np.zeros_like(W)
W_odd_pows = np.zeros_like(W)

# Constants to account for the chemical potential.
# this variable will store exp(k*mnu*beta) for current value of k,
# this is a coefficient in front of the Gibbs state with temperature k*beta
exp_k_mu_beta = np.exp(mu*beta)
exp_mu_beta = np.exp(mu*beta)

# Order in taylor series
k = 0

# Loop over each term in taylor expansion until convergence
while True:
    k += 1

    # Propagate to get the nested terms in taylor expansion
    for _ in xrange(BetaIterSteps):
        # p x -> theta x
        W = fftpack.fft(W, axis=0, overwrite_x=True)
        W *= expV
        # theta x  ->  p x
        W = fftpack.ifft(W, axis=0, overwrite_x=True)

        # p x  ->  p lambda
        W = fftpack.fft(W, axis=1, overwrite_x=True)
        W *= expK
        # p lambda  ->  p x
        W = fftpack.ifft(W, axis=1, overwrite_x=True)

    # normalization needs to be done only for the first term
    if k == 1:
        W /= W.real.sum()*dX*dP
        # Save this Gibbs state for subsequent visualization
        W_Gibbs = np.copy(W.real)

    #################################################################################
    #
    # Perform consistency checks that the normalized W is a physically valid state
    #
    #################################################################################

    norm = W.real.sum()*dX*dP

    # check purity is not larger unity
    if 2.*np.pi*np.sum(W.real**2)*dP*dX / norm**2 > 1:
        print("Warning: Purity cannot be larger than one. Calculation was interrupted.")
        break

    # standard deviations squared
    sigma_p_2 = np.sum(W.real*P**2)*dX*dP / norm - (np.sum(W.real*P)*dX*dP / norm)**2
    sigma_x_2 = np.sum(W.real*X**2)*dX*dP / norm - (np.sum(W.real*X)*dX*dP / norm)**2
    if sigma_x_2 * sigma_p_2 < 0.25:
        print("Warning: Uncertainty principle cannot be violated. Calculation was interrupted.")
        break

    # calculate energy
    current_energy = np.sum(W.real*Hamiltonian)*dP*dX / norm
    if current_energy > previous_energy:
        print("Warning: Ennergy cannot rise. Calculation was interrupted.")
        break
    else:
        previous_energy = current_energy

    #################################################################################
    #
    # Save the results of calculations
    #
    #################################################################################

    # update the value of the coefficient in front of the Gibbs state with temperature k*beta
    exp_k_mu_beta *= exp_mu_beta

    # Add correction into the corresponding
    if k % 2 == 0:
        W_even_pows += exp_k_mu_beta * W
    else:
        W_odd_pows += exp_k_mu_beta * W

    # check for convergence
    peak_val = exp_k_mu_beta * np.linalg.norm(np.ravel(W), np.inf) # maximum value of Wigner func
    print("Iteration %d, norm %.4f, Wigner func peak value %.5f" % (k, norm, peak_val))
    if norm < 1e-4:
        # result converged
        break

#################################################################################
#
# Converting result of propagation into Thomas-Fermi and Bose-Einstein statistics
#
#################################################################################

# Thomas-Fermi distribution
W_TF = np.real(W_odd_pows - W_even_pows)
W_TF /= W_TF.sum()*dX*dP

# Bose-Einstein distribution
W_BE = np.real(W_odd_pows + W_even_pows)
W_BE /= W_BE.sum()*dX*dP

#################################################################################
#
#   Plot the comparison
#
#################################################################################

def PlotWigner(W):
    """
    Plot the Wigner function
    """
    W = fftpack.fftshift(W)

    # Generate Wigner color map
    global_color_max = 0.15  # Maximum value used to select the color range
    global_color_min = 0  #

    extent = [-X_amplitude, X_amplitude - dX, -P_amplitude, P_amplitude - dP]

    plt.imshow(W, origin='lower', extent=extent,
               vmin= global_color_min, vmax=global_color_max, cmap='Reds')
    plt.xlabel('$x$ (a.u.)')
    plt.colorbar(pad=0, shrink=0.5)

plt.subplot(131)
PlotWigner(W_Gibbs)
plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(a)', color='k', fontsize=15)
plt.title("Gibbs Wigner function")
plt.ylabel('$p$ (a.u.)')

plt.subplot(132)
PlotWigner(W_BE)
plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(b)', color='k', fontsize=15)
plt.title("Bose-Einstein Wigner function")

plt.subplot(133)
PlotWigner(W_TF)
plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(c)', color='k', fontsize=15)
plt.title("Thomas-Fermi Wigner function")

plt.show()
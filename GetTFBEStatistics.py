"""
Stand alone script to calculate the Wigner functions of
Thomas-Fermi and Bose-Einstein statistics
"""
import numpy as np
import scipy.fftpack as fftpack
from scipy.linalg import expm, solve
import matplotlib.pyplot as plt

# Specify Hamiltonian of the system:
#  This More potential was takne from the paper http://dx.doi.org/10.1103/PhysRevLett.114.050401
V = lambda x: 15.*(np.exp(-0.5*(x-disp)) - 2.*np.exp(-0.25*(x-disp)))
K = lambda p: 0.5*p**2

# Chemical potential
mu = 0

# Inverse temperature for final state
beta = 5.

####################################################
#
# Parameters of for both numerical algorithms
# (to find Gibbs state and to propagate Moyal equation)
#
####################################################

X_gridDIM = 512       # Discretization grid size in X
P_gridDIM = 512         # Discretization grid size in P

X_amplitude = 8.         # Window range -X_amplitude to X_amplitude
P_amplitude = 9.        # Window range -P_amplitude to P_amplitude

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
# Position the potential such that numerical simulations are balanced
#
#################################################################################


def obj(new_disp):
    """
    Objective function to find the displacement
    """
    potential = V(X_range - new_disp)
    min_indx = np.argmin(potential)
    return np.abs(potential[:min_indx].sum() - potential[min_indx:].sum())

# First asssume no displacement
disp = 0
from scipy.optimize import minimize_scalar

# find new displacement
disp = minimize_scalar(obj).x

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

# Order in taylor series
k = 0

# Loop over each term in taylor expansion until convergence
while True:
    k += 1

    # Propagate to get the nest term in taylor expansion
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

    # Add correction into the corresponding
    if k % 2 == 0:
        W_even_pows += W
    else:
        W_odd_pows += W

    # check for convergence
    peak_val = np.linalg.norm(np.ravel(W), np.inf) # maximum value of Wigner func
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
# Verify that the obtained state is stationary w.r.t. the Moyal equation
#
#################################################################################

def MoyalPropagation(W):
    """
    Propagate wigner function W by the Moyal equation.
    This function is used to verify that the obtained wigner functions
    are steady state solutions of the Moyal equation.
    """
    dt = 0.005 # time increment
    TIterSteps = 500

    # Make copy
    W = np.copy(W)

    # Pre-calculate exp
    expIV = np.exp(-1j*dt*(V(X - 0.5*Theta) - V(X + 0.5*Theta)))
    expIK = np.exp(-1j*dt*(K(P + 0.5*Lambda) - K(P - 0.5*Lambda)))

    for _ in xrange(TIterSteps):
        # p x -> theta x
        W = fftpack.fft(W, axis=0, overwrite_x=True)
        W *= expIV
        # theta x  ->  p x
        W = fftpack.ifft(W, axis=0, overwrite_x=True)

        # p x  ->  p lambda
        W = fftpack.fft(W, axis=1, overwrite_x=True)
        W *= expIK
        # p lambda  ->  p x
        W = fftpack.ifft(W, axis=1, overwrite_x=True)

        # normalization
        W /= W.real.sum()*dX*dP

    return W.real

#################################################################################
#
#   Plot the comparison
#
#################################################################################

def PlotWigner(W):
    #
    W = np.abs(W.real)
    W /= W.max()
    W = np.log10(W)
    #cut_off = -16
    cut_off = np.floor(W.min()) + 1
    W[np.nonzero(W < cut_off)] = cut_off

    extent = [-X_amplitude, X_amplitude - dX, -P_amplitude, P_amplitude - dP]
    plt.imshow(fftpack.fftshift(W), extent=extent, origin='lower', interpolation='nearest')
    plt.colorbar(ticks=[W.min(), 0.5*(W.min() + W.max()), W.max()], shrink=0.9)


plt.subplot(221)
PlotWigner(W_BE)
plt.title("Bose-Einstein (BE) Wigner function")

plt.subplot(222)
PlotWigner(W_TF)
plt.title("Thomas-Fermi (TF) Wigner function")

plt.subplot(223)
plt.semilogy(X_range, fftpack.fftshift(W_BE).sum(axis=0)*dP, '-r', label='Wigner BE')
plt.title('BE')

plt.subplot(224)
plt.semilogy(X_range, fftpack.fftshift(W_TF).sum(axis=0)*dP, '-r', label='Wigner TF')
plt.title('TF')

plt.show()


"""
def PlotWigner(W, global_color_min, global_color_max):
    "
    Plot the Wigner function
    "
    W = fftpack.fftshift(W)

    # Generate Wigner color map
    global_color_max = W.max()  # Maximum value used to select the color range
    global_color_min = W.min()  #

    # adjust minimum bound so that it is at least 2% of the max
    global_color_min = min(global_color_min, -0.02*abs(global_color_max))

    zero_position = abs(global_color_min) / (abs(global_color_max) + abs(global_color_min))
    wigner_cdict = {'red' 	:   ((0., 0., 0.),
                                  (zero_position, 1., 1.),
                                  (1., 1., 1.)),
                    'green' :	((0., 0., 0.),
                                  (zero_position, 1., 1.),
                                  (1., 0., 0.)),
                    'blue'	:	((0., 1., 1.),
                                    (zero_position, 1., 1.),
                                    (1., 0., 0.))
                    }
    wigner_cmap = plt.cm.colors.LinearSegmentedColormap('wigner_colormap', wigner_cdict, 1024)

    extent = [-X_amplitude, X_amplitude - dX, -P_amplitude, P_amplitude - dP]

    plt.imshow(W, origin='lower', extent=extent,
               vmin= global_color_min, vmax=global_color_max, cmap=wigner_cmap)
    plt.xlabel('$x$ (a.u.)')
    plt.colorbar(pad=0, shrink=0.9)

global_color_min = W_TF.min()
global_color_max = W_TF.max()

plt.subplot(231)
PlotWigner(W_BE)
plt.title("Bose-Einstein (BE) Wigner function")
#plt.title("Log plot of Wigner function $\\left(\\log|W_{xp}|\\right)$")
#plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(a)', color='white', fontsize=15)
plt.ylabel('$p$ (a.u.)')

plt.subplot(232)
PlotWigner(W_TF)
plt.title("Thomas-Fermi (TF) Wigner function")

plt.subplot(233)
PlotWigner(W_Gibbs)
plt.title("Gibbs Wigner function")

plt.subplot(234)
PlotWigner(MoyalPropagation(W_BE))
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$p$ (a.u.)')
plt.title("BE Wigner function after Moyal propagation")
#plt.title("Log plot of Wigner function propagated via Moyal equation $\\left(\\log|W_{xp}|\\right)$")
#plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(b)', color='white', fontsize=15)

plt.subplot(235)
PlotWigner(MoyalPropagation(W_TF))
plt.xlabel('$x$ (a.u.)')
plt.title("TF Wigner function after Moyal propagation")

plt.subplot(236)
PlotWigner(MoyalPropagation(W_Gibbs))
plt.xlabel('$x$ (a.u.)')
plt.title("Gibbs Wigner function after Moyal propagation")

plt.show()
"""
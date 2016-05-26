"""
Stand alone script to calculate the Wigner fucntions of ground and first exited states.
"""
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# Specify Hamiltonian of the system
V = lambda x: -0.05*x**2 + 0.03*x**4
K = lambda p: 0.5*p**2

####################################################
#
# Parameters of for both numerical algorithms
# (to find Gibbs state and to propagate Moyal equation)
#
####################################################

X_gridDIM = 512     # Discretization grid size in X
P_gridDIM = 512     # Discretization grid size in P

X_amplitude = 6.     # Window range -X_amplitude to X_amplitude
P_amplitude = 6.     # Window range -P_amplitude to P_amplitude

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

##################################################################
#
# Find energies of the systems by
# diagonalizing the Hamiltonian in coordinate basis
#
##################################################################

# Construct the Hamiltonian in the coordinate basis
Hamiltonian_XBasis = fftpack.fft(np.diag(K(2*np.pi*fftpack.fftfreq(X_gridDIM, dX))), axis=1, overwrite_x=True)
Hamiltonian_XBasis = fftpack.ifft(Hamiltonian_XBasis, axis=0, overwrite_x=True)
Hamiltonian_XBasis += np.diag(V(X_range))

print("Eigenvalues of Hamiltonian %s\n" % np.linalg.eigvalsh(Hamiltonian_XBasis)[:2].real)

##################################################################
#
# Get the ground state Wigner function
#
##################################################################

print("\n\nConstructing Wigner function of ground state\n")

dbeta = 1. # 1/(kT) increment
BetaIterSteps = 1500 # Number of iterations

Hamiltonian = K(P) + V(X)

# Pre-calculate exp
expV = np.exp(-0.5*dbeta*(V(X - 0.5*Theta) + V(X + 0.5*Theta)))
expK = np.exp(-0.5*dbeta*(K(P + 0.5*Lambda) + K(P - 0.5*Lambda)))

# Initialize Wigner function
W = np.ones((P.size, X.size), dtype=np.complex)

previous_energy = np.inf

for Index in xrange(1, BetaIterSteps+1):
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

    # normalization
    W /= W.real.sum()*dX*dP

    if Index % 50 == 0:
        # calculate purity
        current_purity = 2.*np.pi*np.sum(W.real**2)*dP*dX

        # calculate energy
        current_energy = np.sum(W.real*Hamiltonian)*dP*dX

        # standard deviations squared
        sigma_p_2 = np.sum(W.real*P**2)*dX*dP - (np.sum(W.real*P)*dX*dP)**2
        sigma_x_2 = np.sum(W.real*X**2)*dX*dP - (np.sum(W.real*X)*dX*dP)**2

        # test physicality of the state
        try:
            if current_purity > 1.:
                # print("Warning: Purity cannot be larger than one")
                raise RuntimeError

            if previous_energy < current_energy:
                # print("Warning: Energy cannot increase")
                raise RuntimeError

            if sigma_x_2 * sigma_p_2 < 0.25:
                print ("Warning: Uncertainty principle cannot be violated")
                raise RuntimeError

            # since the current wigner function is physical then save a copy
            W_old = np.copy(W.real)

            print('Purity %.15f\t Energy %.15f' % (current_purity, current_energy))

            previous_energy = current_energy
        except RuntimeError:
            # decrease temperature step by half
            dbeta *= 0.5
            # that corresponds to square root of pre-calculated exp
            np.sqrt(expV, out=expV)
            np.sqrt(expK, out=expK)
            # revert to older state
            W[:] = W_old

# Save a copy of the ground state Wigner function
W_ground = np.copy(W.real)

##################################################################
#
# Get the first exited state Wigner function
#
##################################################################

print("\n\nConstructing Wigner function of first excited state\n")

dbeta = 0.3 # 1/(kT) increment
BetaIterSteps = 7000 # Number of iterations

# Pre-calculate exp
expV = np.exp(-0.5*dbeta*(V(X - 0.5*Theta) + V(X + 0.5*Theta)))
expK = np.exp(-0.5*dbeta*(K(P + 0.5*Lambda) + K(P - 0.5*Lambda)))

# Initialize Wigner function
W = np.ones((P.size, X.size), dtype=np.complex)

previous_energy = np.inf

for Index in xrange(1, BetaIterSteps+1):
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

    # normalization
    W /= W.real.sum()*dX*dP

    # Essential part: Project out the ground state
    proj = 2.*np.pi*np.sum(W.real * W_ground)*dX*dP
    W -= proj * W_ground

    # normalization
    W /= W.real.sum()*dX*dP

    if Index % 50 == 0:
        # calculate purity
        current_purity = 2.*np.pi*np.sum(W.real**2)*dP*dX

        # calculate energy
        current_energy = np.sum(W.real*Hamiltonian)*dP*dX

        # standard deviations squared
        sigma_p_2 = np.sum(W.real*P**2)*dX*dP - (np.sum(W.real*P)*dX*dP)**2
        sigma_x_2 = np.sum(W.real*X**2)*dX*dP - (np.sum(W.real*X)*dX*dP)**2

        # test physicality of the state
        try:
            if current_purity > 1.:
                #print("Warning: Purity cannot be larger than one")
                raise RuntimeError

            if previous_energy < current_energy:
                #print("Warning: Energy cannot increase")
                raise RuntimeError

            if sigma_x_2 * sigma_p_2 < 0.25:
                #print ("Warning: Uncertainty principle cannot be violated")
                raise RuntimeError

            # since the current wigner function is physical then save a copy
            W_old = np.copy(W.real)

            print('Purity %.15f\t Energy %.15f' % (current_purity, current_energy))

            previous_energy = current_energy

        except RuntimeError:
            # decrease temperature step by half
            dbeta *= 0.5
            # that corresponds to square root of pre-calculated exp
            np.sqrt(expV, out=expV)
            np.sqrt(expK, out=expK)
            # revert to older state
            W[:] = W_old

# Save a copy of the first excited state
W_exited1 = np.copy(W.real)

##################################################################
#
# Plot obtained states
#
##################################################################

def MoyalPropagation(W):
    """
    Propagate wigner function W by the Moyal equation.
    This function is used to verify that the obtained wigner functions
    are steady state solutions of the Moyal equation.
    """
    # Make a copy
    W = np.copy(W)

    dt = 0.005 # time increment
    TIterSteps = 2000

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

    return fftpack.fftshift(W.real)


def PlotWigner(W):
    """
    Plot the Wigner function
    """
    print("\nMoyal equation propagation\n")
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

##################################################################

plt.subplot(221)
PlotWigner(W_ground)
#plt.title("Ground state\nWigner function")
plt.text(0.6*X_amplitude, -0.8*P_amplitude, '(a)', fontsize=15)
plt.ylabel('$p$ (a.u.)')

plt.subplot(222)
PlotWigner(W_exited1)
#plt.title("First exited state\nWigner function")
plt.text(0.6*X_amplitude, -0.8*P_amplitude, '(b)', fontsize=15)

def MarginalPlot(W):
    plt.semilogy(X_range, fftpack.fftshift(W).sum(axis=0)*dP, '-r', label='before propagation')
    plt.semilogy(X_range, MoyalPropagation(W).sum(axis=0)*dP, '--b', label='after propagation')

    plt.legend(loc='lower center', fontsize=10)
    plt.xlabel('$x$ (a.u.)')
    plt.ylim((1e-16, 1e0))
    # plt.ylabel('Marginal coordinate distribution, $\\int W_{xp}dp$')

plt.subplot(223)
MarginalPlot(W_ground)
# plt.title('Verification of ground state Wigner function')
plt.text(0.7*X_amplitude, 1e-3, '(c)', color='k', fontsize=15)
plt.ylabel('Marginal coordinate distribution')

plt.subplot(224)
MarginalPlot(W_exited1)
# plt.title('Verification of first exited state Wigner function')
plt.text(0.7*X_amplitude, 1e-3, '(d)', color='k', fontsize=15)

plt.show()

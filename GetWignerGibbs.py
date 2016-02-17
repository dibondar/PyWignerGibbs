"""
Stand alone script to calculate the Gibbs state
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

X_gridDIM = 512       # Discretization grid size in X
P_gridDIM = 512         # Discretization grid size in P

X_amplitude = 10         # Window range -X_amplitude to X_amplitude
P_amplitude = 10        # Window range -P_amplitude to P_amplitude

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

####################################################
#
# Get Gibbs state
#
####################################################

dbeta = 0.005 # 1/(kT) increment
BetaIterSteps = 200 # Number of iterations

# Pre-calculate exp
expV = np.exp(-0.5*dbeta*(V(X - 0.5*Theta) + V(X + 0.5*Theta)))
expK = np.exp(-0.5*dbeta*(K(P + 0.5*Lambda) + K(P - 0.5*Lambda)))

# Initialize Wigner function
W = np.ones((P.size, X.size), dtype=np.complex)

for Index in xrange(BetaIterSteps):
    if Index%100 == 0:
        print('\tBeta iteration num  %d' % Index)

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
    W /= W.sum()*dX*dP

# Save Gibbs state
GibbsW = np.copy(W)

####################################################
#
# Verify Gibbs state. Check that it is a stationary
# state of the Moyal equation
#
####################################################

dt = 0.005 # time increment
TIterSteps = 300

# Pre-calculate exp
expIV = np.exp(-1j*dt*(V(X - 0.5*Theta) - V(X + 0.5*Theta)))
expIK = np.exp(-1j*dt*(K(P + 0.5*Lambda) - K(P - 0.5*Lambda)))

for Index in xrange(TIterSteps):
    if Index%100 == 0:
        print('\tTime iteration num  %d' % Index)

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
    W /= W.sum()*dX*dP

####################################################
#
#   Plot the comparison
#
####################################################

print "Difference between initial and final state ", np.abs(W - GibbsW).sum()*dX*dP

def PlotWigner(W):
    #
    W = np.abs(W.real)
    W /= W.max()
    W = np.log10(W)
    cut_off = -14
    W[np.nonzero(W < cut_off)] = cut_off

    extent = [-X_amplitude, X_amplitude - dX, -P_amplitude, P_amplitude - dP]
    plt.imshow(fftpack.fftshift(W), extent=extent, origin='lower', interpolation='nearest')
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.colorbar(ticks=[W.min(), 0.5*(W.min() + W.max()), W.max()])

plt.subplot(211)
PlotWigner(GibbsW)
#plt.title("Log plot of Gibbs state Wigner function $\\left(\\log|W_{xp}|\\right)$")
plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(a)', color='white', fontsize=15)

plt.subplot(212)
PlotWigner(W)
#plt.title("Log plot of Wigner function of \nGibbs state propagated via Moyal equation $\\left(\\log|W_{xp}|\\right)$")
plt.text(0.7*X_amplitude, -0.8*P_amplitude, '(b)', color='white', fontsize=15)
plt.show()







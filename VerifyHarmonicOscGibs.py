"""
Compare numerically constructed Gibbs state of the Harmonic oscilator
with the corresponding analytical solution.
"""
import numpy as np
import scipy.fftpack as fftpack

# Specify Hamiltonian of the system
omega = 3
V = lambda x: 0.5*(omega*x)**2
K = lambda p: 0.5*p**2

def ExactGibbs(x, p, beta):
    """
    Analytical expression for the Gibbs state
    """
    C = np.tanh(0.5*beta*omega)
    return C * np.exp(-2.*C*(V(x)+K(p))/omega) / np.pi

####################################################
#
# Parameters of for numerical algorithm
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
# Get Gibbs state and compare with the analytical expression
#
####################################################

dbeta = 0.005 # 1/(kT) increment
BetaIterSteps = 200 # Number of iterations

# Pre-calculate exp
expV = np.exp(-0.5*dbeta*(V(X - 0.5*Theta) + V(X + 0.5*Theta)))
expK = np.exp(-0.5*dbeta*(K(P + 0.5*Lambda) + K(P - 0.5*Lambda)))

# Initialize Wigner function
W = np.ones((P.size, X.size), dtype=np.complex)

for beta in np.arange(dbeta, dbeta*BetaIterSteps, dbeta):
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

    # get exact Gibbs state
    ExactW = ExactGibbs(X, P, beta)

    print " for beta = %.2e \t Difference between two states %.2e" % \
          (beta, np.linalg.norm(W - ExactW)*dX*dP)
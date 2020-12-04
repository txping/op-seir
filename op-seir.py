import os
import argparse
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from autograd.scipy.integrate import odeint
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Optimal Control for COVID-19')
ap = parser.add_argument
ap('--country', help='country', type=str, default='US', choices=['US','UK','FC','CN'])
ap('--lambda1', help='lambda1', type=float, default=1e-5)
ap('--lambda2', help='lambda2', type=float, default=1e-3)
ap('--p0', help='initialize beta, epsilon, gamma and mu', type=float, default=[0,0.3,0.1,2e-3], nargs='+')
ap('--tau', help='step size for minimization', type=float, default=1e-2)
ap('--d', help='fit data every d days', type=int, default=2)
ap('--h', help='step size for ODE solver', type=float, default=2e-3)
ap('--Ts', help='starting time', type=int, default=0)
ap('--Te', help='end time', type=int, default=30)
ap('--tol', help='error tol', type=float, default=1e-6)
ap('--maxiter', help='max iter', type=int, default=100)
ap('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

country = args.country
lambda1 = args.lambda1
lambda2 = args.lambda2
p0 = args.p0
tau = args.tau
d = args.d # sample data every d days
h = args.h # step size of ODE solver
Ts = args.Ts
Te = args.Te
tol = args.tol
maxiter = args.maxiter

n = int((Te-Ts)/d) # number of intervals
m = int(d/h) # number of steps of ODE solver
t = np.linspace(Ts,Te,n*m+1) # Ts, Te included, n*m+1 points

def load_data(country):
    I_global = pd.read_csv('./data/time_series_covid19_confirmed_global.csv')
    D_global = pd.read_csv('./data/time_series_covid19_deaths_global.csv')

    if country == 'US':
        N0 = 331002651
        I_country = I_global.loc[I_global['Country/Region']=='US'].values.flatten()[4:].astype('float64')
        D_country = D_global.loc[D_global['Country/Region']=='US'].values.flatten()[4:].astype('float64')
    elif country == 'UK':
        N0 = 67886011
        I_country = I_global.loc[I_global['Country/Region']=='United Kingdom'].iloc[-1,4:].values.astype('float64')
        D_country = D_global.loc[D_global['Country/Region']=='United Kingdom'].iloc[-1,4:].values.astype('float64')
    elif country == 'FC':
        N0 = 65273511
        I_country = I_global.loc[I_global['Country/Region']=='France'].iloc[-1,4:].values.astype('float64')
        D_country = D_global.loc[D_global['Country/Region']=='France'].iloc[-1,4:].values.astype('float64')
    else:
        N0 = 1439323776
        I_country = I_global.loc[I_global['Country/Region']=='China'].iloc[:,4:].sum(axis=0).values.astype('float64')
        D_country = D_global.loc[D_global['Country/Region']=='China'].iloc[:,4:].sum(axis=0).values.astype('float64')
    return N0, I_country, D_country

def w_mse(Ip, Dp, Ic, Dc):
    # Ip Ic have n+1 points, two ending points included
    return lambda1 * np.sum((Ip-Ic)**2) + lambda2 * np.sum((Dp-Dc)**2)

def initial_theta(Ic, Dc, p):
    beta = p[0] * np.ones([n,m])
    epsilon = p[1] * np.ones([n,m])
    gamma = p[2] * np.ones([n,m])

    mu = np.zeros([n,m])
    mus = (Dc[1:]-Dc[:-1])/(d*Ic[1:]+1e-24)
    for i in range(len(mus)):
        mu[i] = mus[i] * np.ones(m)

    #mu = p[3] * np.ones([n,m])
    theta = np.array([beta, epsilon, gamma, mu])
    return theta


def forward(U0, theta):
    # solve the forward problem
    S, E, I, R, D = U0
    N = S+E+I+R
    beta, epsilon, gamma, mu = theta
    for i in range(n):
        for k in range(m):
            S[i,k+1] = S[i,k] / (1 + h*beta[i,k]*I[i,k]/N[i,k])
            E[i,k+1] = (E[i,k]+h*beta[i,k]*S[i,k+1]*I[i,k]/N[i,k]) / (1+h*epsilon[i,k])
            I[i,k+1] = (I[i,k]+h*epsilon[i,k]*E[i,k+1]) / (1+h*(gamma[i,k]+mu[i,k]))
            R[i,k+1] = R[i,k] + h*gamma[i,k]*I[i,k+1]
            D[i,k+1] = D[i,k] + h*mu[i,k]*I[i,k+1]
            N[i,k+1] = S[i,k+1] + E[i,k+1] + I[i,k+1] + R[i,k+1]
        if i<n-1:
            S[i+1,0] = S[i,m]
            E[i+1,0] = E[i,m]
            I[i+1,0] = I[i,m]
            R[i+1,0] = R[i,m]
            D[i+1,0] = D[i,m]
            N[i+1,0] = N[i,m]

    U = [S, E, I, R, D]

    return U, N

def backward(V0, theta, U, N, Ip, Dp, Ic, Dc):
    # solve the backward problems
    VS, VE, VI, VR, VD = V0
    beta, epsilon, gamma, mu = theta
    S, E, I, R, D = U

    for i in range(n-1,-1,-1):
        for k in range(m,0,-1):
            VS[i,k-1] = ((VS[i,k]*(N[i,k-1]**2) + h*beta[i,k-1]*I[i,k-1]*(N[i,k-1]-S[i,k-1])*VE[i,k])
                        /((N[i,k-1]**2) + h*beta[i,k-1]*I[i,k-1]*(N[i,k-1]-S[i,k-1])))
            VE[i,k-1] = (VE[i,k] + h*epsilon[i,k-1]*VI[i,k])/(1 + h*epsilon[i,k-1])
            VI[i,k-1] = ((VI[i,k] + h * (gamma[i,k-1]*VR[i,k]
                                      + mu[i,k-1]*VD[i,k]
                                      - beta[i,k-1]*S[i,k-1]*(N[i,k-1]-I[i,k-1])*(VS[i,k-1]-VE[i,k-1])/(N[i,k-1]**2)))
                        /(1+h*(gamma[i,k-1]+mu[i,k-1])))
        if i>0:
            VS[i-1,m] = VS[i,0]
            VE[i-1,m] = VE[i,0]
            VI[i-1,m] = VI[i,0] + 2*lambda1*(Ip[i]-Ic[i])
            VR[i-1,m] = VR[i,0]
            VD[i-1,m] = VD[i,0] + 2*lambda2*(Dp[i]-Dc[i])

    VS =  VS[:,1:]
    VE =  VE[:,1:]
    VI =  VI[:,1:]
    VR =  VR[:,1:]
    VD =  VD[:,1:]

    return VS, VE, VI, VR, VD

def callback(Ic, Dc, I, D, theta, itr):
    beta, epsilon, gamma, mu = theta
    R0 = np.divide(beta, gamma+mu, out=np.zeros_like(beta), where=(gamma+mu)!=0)

    ax_beta.cla()
    ax_beta.plot(t[:-1], beta.flatten(), label='beta')

    ax_epsilon.cla()
    ax_epsilon.plot(t[:-1], epsilon.flatten(), label='epsilon')

    ax_gamma.cla()
    ax_gamma.plot(t[:-1], gamma.flatten(), label='gamma')

    ax_mu.cla()
    ax_mu.plot(t[:-1], mu.flatten(), label='mu')

    ax_R0.cla()
    ax_R0.plot(t[:-1], R0.flatten(), label='mu')

    for ax in [ax_beta, ax_epsilon, ax_gamma, ax_mu]:
        ax.set_xticks(range(Ts,Te+1,d))
        ax.legend(loc='upper left')
        ax.grid()
    ax_mu.set_xlabel('Time [days]')

    ax_I.cla()
    ax_I.plot(range(Ts,Te+1,d), Ic, 'r*', label='I-data')
    ax_I.plot(t[1:], I[:,1:].flatten(), 'b', label='I-SEIRD')
    # real data does not include starting point

    ax_D.cla()
    ax_D.plot(range(Ts,Te+1,d), Dc, 'r*', label='D-data')
    ax_D.plot(t[1:], D[:,1:].flatten(), 'b', label='D-SEIRD')

    for ax in [ax_I, ax_D]:
        ax.set_xticks(range(Ts,Te+1,d))
        ax.legend()
        ax.grid()
    ax_D.set_xlabel('Time [days]')

    fig1.tight_layout()
    fig2.tight_layout()
    plt.savefig('./outputs/png/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)


def minimize(U0, theta, callback, Ic, Dc):

    iter = 0
    theta_ = theta + 0.1
    beta, epsilon, gamma, mu = theta

    while np.linalg.norm(theta_-theta)/np.linalg.norm(theta_) > tol and iter < maxiter:

        U, N = forward(U0, theta)
        S, E, I, R, D = U
        Ip = np.append(I[0,0], I[:,m]) # n+1 points, two ending points included
        Dp = np.append(D[0,0], D[:,m])

        if callback is not None:
            callback(Ic, Dc, I, D, theta, iter)

        V0 = np.zeros([5,n,m+1])
        V0[2,n-1,m] = 2*lambda1*(Ip[n]-Ic[n])
        V0[4,n-1,m] = 2*lambda2*(Dp[n]-Dc[n])
        VS, VE, VI, VR, VD = backward(V0, theta, U, N, Ip, Dp, Ic, Dc)

        S =  S[:,1:]
        E =  E[:,1:]
        I =  I[:,1:]
        R =  R[:,1:]
        D =  D[:,1:]
        N =  N[:,1:]

        beta_ = beta + (tau*100) * (S*I*(VS-VE)/N)
        epsilon_ = epsilon + (tau) * (E*(VE-VI))
        gamma_ = gamma + (tau) * (I*(VI-VR))
        mu_ = mu + (tau/100) * (I*(VI-VD))

        beta_ = np.clip(beta_, 0, 5)
        epsilon_ = np.clip(epsilon_, 0.2, 0.25)
        gamma_ = np.clip(gamma_, 0.1, 0.2)
        mu_ = np.clip(mu_, 0, 0.02)

        theta_ = np.array([beta_, epsilon_, gamma_, mu_])
        theta = np.array([beta, epsilon, gamma, mu])

        loss = w_mse(Ip, Dp, Ic, Dc) + \
            LA.norm(beta-beta_)/(2*tau) + LA.norm(epsilon-epsilon_)/(2*tau) + LA.norm(gamma-gamma_)/(2*tau) + LA.norm(mu-mu_)/(2*tau/100)
        H = -VS*beta*I/N + VE*(beta*S*I/N-epsilon*E) + VI*(epsilon*E-(gamma+mu)*I) + VR*gamma*I + VD*mu*I + \
            LA.norm(beta-beta_)/(2*tau) + LA.norm(epsilon-epsilon_)/(2*tau) + LA.norm(gamma-gamma_)/(2*tau) + LA.norm(mu-mu_)/(2*tau/100)
        print("[{}] loss = {}, H = {}".format(iter, loss, H[1,1]))

        beta = beta_
        epsilon = epsilon_
        gamma = gamma_
        mu = mu_

        iter = iter + 1

    return theta, U

if __name__ == '__main__':

    N0, I_country, D_country = load_data(country)

    # Ic, Dc are data to be fitted, two boundary points included
    Ic, Dc = I_country[Ts:Te+1:d], D_country[Ts:Te+1:d]

    # scheduled control
    #Ic = Ic[0]+0.5*(Ic - Ic[0])
    #Dc = Dc[0]+0.5*(Dc - Dc[0])
    #print('Ic={}, Dc={}'.format(Ic[-1], Dc[-1]))

    if args.resume:
        init_theta = np.load(os.path.join('./outputs/','theta_us_'+str(Ts)+'_'+str(Te)+'.npy'))
    else:
        if p0[0]==0:
            theta_ = np.load(os.path.join('./outputs/','theta_us_'+str(Ts-30)+'_'+str(Te-30)+'.npy'))
            p = theta_[:,-1,-1]
        else:
            p = p0
        init_theta = initial_theta(Ic, Dc, p)

    U0 = np.zeros([5,n,m+1])
    if Ts>0:
        U_ = np.load(os.path.join('./outputs/','U_us_'+str(Ts-30)+'_'+str(Te-30)+'.npy'))
        U0[0,0,0] = U_[0,-1,-1]
        U0[1,0,0] = U_[1,-1,-1]
        U0[2,0,0] = Ic[0]
        U0[3,0,0] = U_[3,-1,-1]
        U0[4,0,0] = Dc[0]
    else:
        U0[0,0,0] = N0-1
        U0[1,0,0] = 1
        U0[2,0,0] = Ic[0]
        U0[3,0,0] = 0
        U0[4,0,0] = Dc[0]

    fig1 = plt.figure(figsize=(8, 8))
    ax_beta = fig1.add_subplot(511)
    ax_epsilon = fig1.add_subplot(512)
    ax_gamma = fig1.add_subplot(513)
    ax_mu = fig1.add_subplot(514)
    ax_R0 = fig1.add_subplot(515)
    plt.show(block=False)

    fig2 = plt.figure(figsize=(8, 8))
    ax_I = fig2.add_subplot(211)
    ax_D = fig2.add_subplot(212)
    plt.show(block=False)

    optimized_theta, U = minimize(U0, init_theta, callback, Ic, Dc)
    np.save(os.path.join('./outputs/','theta_us_'+str(Ts)+'_'+str(Te)+'.npy'), optimized_theta)
    np.save(os.path.join('./outputs/','U_us_'+str(Ts)+'_'+str(Te)+'.npy'), U)

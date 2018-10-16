# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:28:36 2018

@author: gx

(μ/μW, lambda)-CMA-ES
"""
import numpy as np
import math
from scipy.linalg import eig
from numpy.linalg import linalg as LA

def eigendecomposition(x):
#b = np.random.randint(-2000,2000,size=(2,2))
#b = (b + b.T)/2
    u,B=eig(x)
    u=np.real(u)
    D_square=np.diag(u)
    D_halfinverse=np.diag([e**(-0.5) for e in u])
    C_replicate=B@D_square@B.T
    C_halfinverse=B@D_halfinverse@B.T
    return C,C_halfinverse,D_square,B




#problem instance related quantities
n=4
sigma=0.5
mean=np.array([10]*n)
g=[0,]
def stop_criterion(x=g):
    if x[0]>=10: 
        return True
#def f(x): return math.exp(-0.5*(x)**2)/((2*math.pi)**(-0.5))
def f(x): return (3*x**3+6*x**2+2*x**1+10)

#hyper-parameters setting
c_m=1
hp_lambda=4+int(3*np.log(n))
mu=int(hp_lambda/2)
c_1=2/n**2
w_pre=[np.log((hp_lambda+1)/2)-np.log(i) for i in range(1,hp_lambda+1)]
w_pre=np.array(w_pre)
mu_eff=w_pre[0:mu].sum()**2/((w_pre[0:mu]*w_pre[0:mu]).sum())
mu_eff_minus=w_pre[mu:].sum()**2/((w_pre[mu:]*w_pre[mu:]).sum())
c_mu=min(mu_eff/n**2,1-c_1)
#specific issues about negative mu (since 2016), middle quantities
a_mu=1+c_1/c_mu
a_mu_eff=1+2*mu_eff_minus/(mu_eff+2)
a_posdef=(1-c_1-c_mu)/(n*c_mu)
#then assign weights(including positive and negative)
w=[]
w_pre_negativesum=w_pre[mu:].sum()
w_pre_positivesum=w_pre[0:mu].sum()
for i in w_pre:
    if i>=0:
        w_middle1=i/w_pre_positivesum
        w.append(w_middle1)
    elif i<0:
        w_middle2=i*min(a_mu,a_mu_eff,a_posdef)/(-w_pre_negativesum)
        w.append(w_middle2)
    else:
        raise Exception('value exception in w_pre')
w=np.array(w)
if w.size != w_pre.size:
    raise Exception('value assignment exception in w')
#step-size control(constant part)
c_sigma=(mu_eff+2)/(n+mu_eff+5)
d_sigma=c_sigma+1+2*max(0,((mu_eff-1)/(n+1))**(0.5)-1)
#constants for covariance matrix adaptation, a is a middle quantitiy
a_cov=2
c_c=(mu_eff/n+4)/(n+4+2*mu_eff/n)
c_1=a_cov/(mu_eff+(n+1.3)**2)
c_mu=min(1-c_1,a_cov*(mu_eff-2+1/mu_eff)/(a_cov*mu_eff/2+(n+2)**2))
#hyper-parameters setting ends here

#intialization of model
p_sigma=np.array([0]*n)
p_cumulation=np.array([0]*n)
C=np.diag(np.array([1]*n))
#recordings for visualization
f_value_tree=[]
offspring_tree=[]
mean_path=[]
C_path=[]
C_replicate_path=[]
D_path=[]
B_path=[]
sigma_adaptation=[]
h_sigma_selection=[]
p_sigma_path=[]
p_cumulation_path=[]
#session of the evolution process,'ng' means next generation
while stop_criterion() !=True:
    C_replicate,C_halfinverse,D,B=eigendecomposition(C)
    offsprings=np.random.multivariate_normal(mean,C,size=hp_lambda)
    f_value=np.array([f(e).sum() for e in offsprings])
    f_value=np.matrix(f_value)
    offsprings=offsprings[np.array((-f_value[0,:]).argsort())[0],:]
    f_value=np.array(f_value)
    y=(offsprings-mean)/sigma
    y_positivesum=sum(np.array(list(map(lambda a,b:a*b,w[:mu],y[:mu]))))
    mean_ng=mean+c_m*sigma*y_positivesum
    g[0]=g[0]+1
    #step-size control
    p_sigma=(1-c_sigma)*p_sigma+(c_sigma*(2-c_sigma)*mu_eff)**(0.5)*C_halfinverse@y_positivesum
    E_N=(n**(0.5)*(1-1/(4*n)-1/(21*n**2)))
    sigma=sigma*math.exp(c_sigma/d_sigma*(LA.norm(p_sigma)/E_N-1))
    #Covariance-matrix adaptation
    if LA.norm(p_sigma)/(1-(1-c_sigma)**2)<(1.4+2/(n+1))*E_N:
        h_sigma=1
    else:
        h_sigma=0
    p_cumulation=(1-c_c)*p_cumulation+h_sigma*(c_c*(2-c_c)*mu_eff)**(0.5)*y_positivesum
    w_0=w.copy()
    for e,y in zip(w_0,offsprings):
        if e >=0:
            continue
        else:
            e=n/LA.norm(C_halfinverse@y)**2
    delta_h_sigma=min((1-h_sigma)*c_c*(2-c_c),1)
    pcpc_t=np.matrix(p_cumulation).T@np.matrix(p_cumulation)
    weighted_yy_t=sum(np.array(list(map(lambda e,w0:np.matrix(e).T@np.matrix(e),y,w_0))))
    C_ng=(1+c_1*delta_h_sigma-c_1-c_mu*w.sum())*C+c_1*pcpc_t+c_mu*weighted_yy_t
    #main evolution of model
    C=C_ng
    mean=mean_ng
    #recording
    f_value_tree.append(('generation=%d'%g[0],f_value))
    offspring_tree.append(('generation=%d'%g[0],offsprings))
    mean_path.append(('generation=%d'%g[0],mean))
    C_path.append(('generation=%d'%g[0],C))
    C_replicate_path.append(('generation=%d'%(g[0]-1),C_replicate))
    D_path.append(('generation=%d'%g[0],D))
    B_path.append(('generation=%d'%g[0],B))
    sigma_adaptation.append(('generation=%d'%g[0],sigma))
    h_sigma_selection.append(('generation=%d'%g[0],h_sigma))
    p_sigma_path.append(('generation=%d'%g[0],p_sigma))
    p_cumulation_path.append(('generation=%d'%g[0],p_cumulation))































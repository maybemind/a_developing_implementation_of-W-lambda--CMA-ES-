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
import matplotlib.pyplot as plt
import imageio
import os

def eigendecomposition(C):
#b = np.random.randint(-2000,2000,size=(2,2))
#b = (b + b.T)/2
    u,B=eig(C)
    u=np.real(u)
    D_square=np.diag(u)
    D_halfinverse=np.diag([e**(-0.5) for e in u])
    C_replicate=B@D_square@B.T
    C_halfinverse=B@D_halfinverse@B.T
    return C,C_halfinverse,D_square,B

def f_for_plot(x,y):
    return -((x-10)**2+(y-10)**2)

def cotour_plot():
    x=np.linspace(-10,30,num=1000)
    y=np.linspace(-10,30,num=1000)
    X,Y=np.meshgrid(x,y)
    Z=f_for_plot(X,Y)
    plt.contour(X,Y,Z,100,cmap='RdGy')
    cb=plt.colorbar()
    cb.set_label('f_value')
    
#save image
def saveri(sname):
    plt.savefig(str(path)+'\\'+'results'+'\\'+str(sname)+'.png',format='png',dpi=1200)
#save table
def savert(string,name):
    out=open(path+'\\'+'results'+'\\'+str('table')+'.txt','a')
    out.write(str(name))
    out.write('\n')
    out.write(str(string))
    out.write('\n')
    out.close()


def main():
    #problem instance related quantities
    n=2
    sigma=0.5
    mean=np.array([0]*n)
    g=[0,]
    generation_limit=8
    def stop_criterion(x=g):
        if x[0]>=generation_limit: 
            return True
    #def f(x): return math.exp(-0.5*(x)**2)/((2*math.pi)**(-0.5))
    def f(x): return (-(x-10)**2)
    
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
        offsprings=np.random.multivariate_normal(mean,(sigma**2)*C,size=hp_lambda)
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
        for e,y0 in zip(w_0,y):
            if e >=0:
                continue
            else:
                e=e*n/LA.norm(C_halfinverse@y0)**2
        delta_h_sigma=min((1-h_sigma)*c_c*(2-c_c),1)
        pcpc_t=np.matrix(p_cumulation).T@np.matrix(p_cumulation)
        weighted_yy_t=sum(np.array(list(map(lambda e,w0:np.matrix(e).T@np.matrix(e),y,w_0))))
        C_ng=(1+c_1*delta_h_sigma-c_1-c_mu*(w.sum()))*C+c_1*pcpc_t+c_mu*weighted_yy_t
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
    #visualization
    #'tp' means temporary quantity
    def offspring_mean_plot_forgif():
        fig,ax=plt.subplots()
        cotour_plot()
        for j in range(g[0]):
            plt.ylim(-10,30)
            plt.xlim(-10,30)
            ax.scatter(offspring_tree[j][1][:,0],offspring_tree[j][1][:,1],marker='o',color='green')
            for i in range(hp_lambda):
                ax.text(offspring_tree[j][1][:,0][i],offspring_tree[j][1][:,1][i],'%d'%j,color='red')
            mp_tp=np.array([e[1] for e in mean_path[:j+1]])
            plt.plot(mp_tp[:,0],mp_tp[:,1],'-ob')
            saveri('fig%d'%j)
    def offspring_ploter():
        for j in range(g[0]):
            #fig,ax=plt.subplots()
            ax.scatter(offspring_tree[j][1][:,0],offspring_tree[j][1][:,1],marker='o',color='green')
            for i in range(hp_lambda):
                ax.text(offspring_tree[j][1][:,0][i],offspring_tree[j][1][:,1][i],'%d'%j,color='red')
    def mean_path_ploter():
        mp_tp=np.array([e[1] for e in mean_path])
        plt.plot(mp_tp[:,0],mp_tp[:,1],'-ob')
    fig1,ax=plt.subplots()
    cotour_plot()
    mean_path_ploter()
    saveri('fig1')
    fig2,ax=plt.subplots()
    cotour_plot()
    offspring_ploter()
    def rate_ploter():
        sa_tp=np.array([e[1] for e in sigma_adaptation])
        psp_tp=np.array([e[1] for e in p_sigma_path])
        pcp_tp=np.array([e[1] for e in p_cumulation_path])
        hss_tp=np.array([e[1] for e in h_sigma_selection])
        x=np.linspace(1,g[0],num=g[0])
        plt.ylim(0,10)
        plt.plot(x,sa_tp,label='sigma',color='red')
        plt.plot(x,psp_tp,label='p_sigma',color='blue')
        plt.plot(x,pcp_tp,label='p_c',color='green')
        plt.plot(x,hss_tp,label='h',color='black')
        plt.xlabel('generation')
        leg=ax.legend()
    saveri('fig2')
    fig3,ax=plt.subplots()
    rate_ploter()
    saveri('fig3')
    savert(f_value_tree,'f_value_tree')
    savert(offspring_tree,'offspring_tree')
    savert(mean_path,'mean_path')
    savert(C_path,'C_path')
    savert(C_replicate_path,'C_replicate_path')
    savert(D_path,'D_path')
    savert(B_path,'B_path')
    savert(sigma_adaptation,'sigma_adaptation')
    savert(h_sigma_selection,'h_sigma_selection')
    savert(p_sigma_path,'p_sigma_path')
    savert(p_cumulation_path,'p_cumulation_path')
    
    
def im2gif(path,filenames):
    with imageio.get_writer(path+'\\'+'slideshow.gif', mode='I',fps=5) as writer:
        for filename in filenames:
            print(filename)
            image = imageio.imread(path+'\\'+filename)
            writer.append_data(image)
        
if __name__=='__main__':
    path=r'C:\Users\gx\projects\results'
    main()
    l=os.listdir(path)
    im2gif(path,l)
























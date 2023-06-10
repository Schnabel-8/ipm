import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, hessian

import numpy as np

import json as js

import matplotlib as mpl
import matplotlib.pyplot as plt

import os


def IPM(f,cons,A,b,x0,lmd=0.1,gamma=2,epsf=1e-4,eps=1e-4,beta=0.5,tau=0.01):
    data=[list(x0)]
    hashcode=hex(hash(hash(f)+hash(cons)))
    
    cons_num=np.shape(cons(x0))[0]
    A_num=np.shape(A)[0]
    x_num=np.shape(x0)[0]
    y=np.hstack((x0,lmd*np.ones(cons_num),lmd*np.ones(A_num)))
    
    # compute hessian and jacobi
    f_hess=hessian(f)
    f_jac=jacfwd(f)
    cons_jac=jacfwd(cons)
    cons_jac_mat=lambda x:np.vstack(cons_jac(x))
    cons_hess=hessian(cons)
    if cons_num>1:
        cons_hess_sum=lambda x,lmd:np.average(cons_hess(x),axis=0,weights=lmd)*(sum(lmd)/cons_num)
    else:
        cons_hess_sum=lambda x,lmd:lmd*cons_hess(x)[0]
    
    t=1
    itr=1
    #while 1:
    while 1:
        # variables
        x=y[0:x_num]
        lmd=y[x_num:x_num+cons_num]
        v=y[x_num+cons_num:]

        # assemble matrix
        r_dual=f_jac(x)+np.matmul(lmd,cons_jac_mat(x))+np.matmul(v,A)
        r_cent=-lmd*cons(x)-1/t
        r_pri=np.matmul(A,np.transpose(x))-b
        
        RH=-np.hstack((r_dual,r_cent,r_pri))
        RH_norm=np.linalg.norm(RH)
        
        lh11=f_hess(x)+cons_hess_sum(x,lmd)
        lh12=np.transpose(cons_jac_mat(x))
        lh13=np.transpose(A)
        lh21=-np.transpose(lh12*lmd)
        lh22=-np.diag(cons(x))
        
        LH1=np.hstack((lh11,lh12,lh13))
        LH2=np.hstack((lh21,lh22,np.zeros((cons_num,A_num))))
        LH3=np.hstack((A,np.zeros((A_num,cons_num+A_num))))
        
        LH=np.vstack((LH1,LH2,LH3))
        dy=np.linalg.solve(LH,RH)
        # line search
        dlmd=dy[x_num:x_num+cons_num]
        where_dlmd=np.where(dlmd>=0)
        divide=-np.delete(lmd,where_dlmd)/np.delete(dlmd,where_dlmd)
        alf_max=1
        if np.size(divide):
            alf_max=min(1,min(divide))
        alf=alf_max*0.99
        while 1:
            if (sum(np.array(cons(y+alf*dy))>=0)==0):
                break
            alf=alf*beta
        while 1:
            y_tmp=y+alf*dy
            
            x_tmp=y_tmp[0:x_num]
            lmd_tmp=y_tmp[x_num:x_num+cons_num]
            v_tmp=y_tmp[x_num+cons_num:]
            
            wr_dual=f_jac(x_tmp)+np.matmul(lmd_tmp,cons_jac_mat(x_tmp))+np.matmul(v_tmp,A)
            wr_cent=-lmd_tmp*cons(x_tmp)-1/t
            wr_pri=np.matmul(A,np.transpose(x_tmp))-b     
            wRH=-np.hstack((wr_dual,wr_cent,wr_pri))
            wRH_norm=np.linalg.norm(wRH)
            if(wRH_norm<=(1-tau*alf)*RH_norm):
                break
            alf=alf*beta
        y=y+alf*dy
        
        # variables
        x=y[0:x_num]
        lmd=y[x_num:x_num+cons_num]
        v=y[x_num+cons_num:]
        
        data.append(list(x))
        
        eta=-np.dot(cons(x),lmd)
        # assemble matrix
        r_dual=f_jac(x)+np.matmul(lmd,cons_jac_mat(x))+np.matmul(v,A)
        r_pri=np.matmul(A,np.transpose(x))-b
        
        norm_dual=np.linalg.norm(r_dual)
        norm_pri=np.linalg.norm(r_pri)
        
        if((norm_dual<=epsf)&(norm_pri<=epsf)&(eta<=eps)):
            break
        print("iter: ",itr,"x = ",x,"f= ",f(x),"t= ",t)
        itr=itr+1
        t=gamma*cons_num/eta

    if not os.path.exists("cache"):
        os.mkdir("cache")
    with open('./cache/'+str(hashcode)+'.json', 'w') as f:
        js.dump(data, f)
    return [x,hashcode]


def myplot(f,hashcode,xl=0,xr=1,yl=0,yr=1):
    fd=open("./cache/"+str(hashcode)+".json")
    data=js.load(fd)
    len_d=len(data)
    xd=[]
    yd=[]
    for i in range(0,len_d):
        xd.append(data[i][0])
        yd.append(data[i][1])
    
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 

    stepx = 0.01; stepy = 0.01 

    x = np.arange(xl,xr,stepx); y = np.arange(yl,yr,stepy) 

    X,Y = np.meshgrid(x,y) 

    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(8,8),dpi=100)

    CS = ax.contourf(X, Y, Z,cmap=mpl.cm.rainbow)
    plt.colorbar(CS)

    CS = ax.contour(X, Y, Z)

    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    
    plt.xticks(np.arange(xl,xr,0.1))
    plt.yticks(np.arange(yl,yr,0.1))
    
    plt.plot(xd,yd,color='yellow',marker='o', markerfacecolor='white')
    
    plt.tight_layout()
    plt.show()

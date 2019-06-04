# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:03:42 2019

@author: matdy
"""

import numpy as np
import math
from scipy import sparse
import matplotlib.pyplot as plt
from numpy import matlib
from scipy.sparse import linalg

r=0.01
tau = 0.25
sigma = 0.2
gamma=3
mu = 0.07
beta=0.01
nz = 101
zlow=0.3
zup=0.7
nb = 111
blow = 0
bup = 1.1
PN = 1e6
Tol = 1e-9
maxcountN = 100

sgs = sigma**2
xs=(mu-r)/gamma/sigma**2
eps=math.sqrt(2*r*tau*xs/gamma/sigma**2)
kbar=beta/gamma-(1-gamma)/gamma*(r+(mu-r)**2/2/gamma/sigma**2)
ko=beta/gamma-(1-gamma)/gamma*(r*(1-tau)+(mu-r)**2/2/gamma/sigma**2)
maximum_loss=gamma/(1-gamma)*math.log(ko/kbar)
r=r*(1-tau)
t1=-6.237683

dz = (zup-zlow)/(nz-1)
db = (bup-blow)/(nb-1)
nzb = nz*nb
ib1 = round((1-blow)/db)
dzz = dz**2
dbb = db**2
dbz = db*dz

z_grid=np.linspace(zlow,zup,nz)
b_grid=np.linspace(blow,bup,nb)
b=matlib.repmat(b_grid,1,nz).reshape((nzb,))
z=matlib.repmat(z_grid,nb,1).transpose().reshape((nzb,))

coef_b0z1 = np.zeros(nzb)
coef_b01z = np.zeros(nzb)
coef_b1z0 = np.zeros(nzb)
coef_1bz0 = np.zeros(nzb)
coef_b0z0 = np.zeros(nzb)
coef_b1z1 = np.zeros(nzb)
coef_1bz1 = np.zeros(nzb)
coef_1b1z = np.zeros(nzb)
coef_b11z = np.zeros(nzb)
f = np.zeros(nzb)

czz=0.5*sgs*z**2*(1-z)**2
cbb=0.5*sgs*b**2
cbz=-1*sgs*b*z*(1-z)
cz=((mu-r-gamma*sgs*z)*(1-z)+r*tau/(1-tau)*b*z)*z
cb=(-1*(1-gamma)*sgs*z+sgs-mu)*b
cu=np.zeros(nzb)
c0=(mu-r-r*tau/(1-tau)*b)*z-0.5*sgs*gamma*z**2+r-beta/(1-gamma)
a=(gamma-1)/gamma
czbuy=z
cbbuy=1-b
czsell=-1*np.ones(nzb)

inner_pos=np.arange(nb,nzb-nb)
index=np.where((inner_pos+1)%nb==1)
inner_pos=np.delete(inner_pos,index)
index=np.where((inner_pos+1)%nb==0)
inner_pos=np.delete(inner_pos,index)

coef_b0z1[inner_pos]=czz[inner_pos]/dzz+cz[inner_pos]*(cz[inner_pos]>0)/dz
-cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)+cbz[inner_pos]*(cbz[inner_pos]<0)/(2*dbz)

coef_b01z[inner_pos]=czz[inner_pos]/dzz-cz[inner_pos]*(cz[inner_pos]<0)/dz
-cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)+cbz[inner_pos]*(cbz[inner_pos]<0)/(2*dbz)

coef_b1z0[inner_pos]=cbb[inner_pos]/dbb+cb[inner_pos]*(cb[inner_pos]>0)/db
-cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)+cbz[inner_pos]*(cbz[inner_pos]<0)/(2*dbz)

coef_1bz0[inner_pos]=cbb[inner_pos]/dbb-cb[inner_pos]*(cb[inner_pos]>0)/db
-cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)+cbz[inner_pos]*(cbz[inner_pos]<0)/(2*dbz)

coef_b1z1[inner_pos]=cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)

coef_1b1z[inner_pos]=cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)

coef_b11z[inner_pos]=-1*cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)

coef_1bz1[inner_pos]=-1*cbz[inner_pos]*(cbz[inner_pos]>0)/(2*dbz)

coef_b0z0[inner_pos]=cu[inner_pos]-coef_b0z1[inner_pos]-coef_b01z[inner_pos]
-coef_b1z0[inner_pos]-coef_1bz0[inner_pos]-coef_b1z1[inner_pos]-coef_b11z[inner_pos]
-coef_1b1z[inner_pos]-coef_1bz1[inner_pos]

pos_blow=nb*np.arange(1,nz-1)
pos_bup=nb*np.arange(2,nz)-1
pos_zlow=np.arange(1,nb-1)
pos_zup=np.arange(nzb-nb+1,nzb)

u = np.zeros(nzb)
#u = -6.25*np.ones(nzb)
#u[:] = maximum_loss/2
count=0
while True:
    u_old=u
    exp=np.exp(u)
    count+=1
    Ibuy=np.zeros(nzb)
    Isell=np.zeros(nzb)
    ncoef_b0z1 = np.zeros(nzb)
    ncoef_b01z = np.zeros(nzb)
    ncoef_b1z0 = np.zeros(nzb)
    ncoef_1bz0  = np.zeros(nzb)
    ncoef_b0z0 = np.zeros(nzb)
    buym=np.zeros(nzb)
    sellm=np.zeros(nzb)
    #interior poinr
    ztemp0 = (u[inner_pos+nb]-u[inner_pos-nb])/dz/2
    ztemp1 = (u[inner_pos]-u[inner_pos-nb])/dz
    ztemp2 = (u[inner_pos+nb]-u[inner_pos])/dz
    btemp0 = (u[inner_pos+1]-u[inner_pos-1])/db/2
    btemp1 = (u[inner_pos]-u[inner_pos-1])/db
    btemp2 = (u[inner_pos+1]-u[inner_pos])/db
    #nonlinear term from consumption
    f_0=exp[inner_pos]*(1-z[inner_pos]*ztemp2)
    nc0 = ko*(1/(1-gamma)*(np.abs(f_0)**a)-(np.abs(f_0)**(a-1))*exp[inner_pos]
    +(np.sign(f_0)*np.abs(f_0)**a)*u[inner_pos])
    ncz = ko*((np.abs(f_0)**(a-1))*exp[inner_pos]*z[inner_pos])
    ncu = -1*ko*(np.abs(f_0)**a)
    
    #Non linear part from uz^2, ub^2, ub*uz
    ncz_zz= 2*czz[inner_pos]*(1-gamma)*(czz[inner_pos]*(1-gamma)>0)*ztemp2 
    + 2*czz[inner_pos]*(1-gamma)*(czz[inner_pos]*(1-gamma)<0)*ztemp1
    
    nc0_zz = -1*czz[inner_pos]*(1-gamma)*(czz[inner_pos]*(1-gamma)>0)*ztemp2**2
    + -1*czz[inner_pos]*(1-gamma)*(czz[inner_pos]*(1-gamma)<0)*ztemp1**2
    
    ncb_bb= 2*cbb[inner_pos]*(1-gamma)*(cbb[inner_pos]*(1-gamma)>0)*btemp2 
    +2*cbb[inner_pos]*(1-gamma)*(cbb[inner_pos]*(1-gamma)<0)*btemp1
    
    nc0_bb = -1*cbb[inner_pos]*(1-gamma)*(cbb[inner_pos]*(1-gamma)>0)*btemp2**2
    -cbb[inner_pos]*(1-gamma)*(cbb[inner_pos]*(1-gamma)<0)*btemp1**2
    
    ncz_bz= cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)>0)*btemp2 
    + cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)<0)*btemp1
    
    ncb_bz= cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)>0)*(btemp2>0)*ztemp2 
    + cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)>0)*(btemp2<0)*ztemp1 
    + cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)<0)*(btemp1<0)*ztemp2 
    + cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)<0)*(btemp1<0)*ztemp1
    
    nc0_bz= -1*cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)>0)*btemp2*(btemp2>0)*ztemp2 
    - cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)>0)*btemp2*(btemp2<0)*ztemp1 
    - cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)<0)*btemp1*(btemp1<0)*ztemp2 
    - cbz[inner_pos]*(1-gamma)*(cbz[inner_pos]*(1-gamma)<0)*btemp1*(btemp1>0)*ztemp1
    
    # trading condition
    buym[inner_pos] = czbuy[inner_pos]*ztemp2 + (cbbuy[inner_pos]>0)*cbbuy[inner_pos]*btemp2
    +(cbbuy[inner_pos]<0)*cbbuy[inner_pos]*btemp1
    
    sellm[inner_pos] = czsell[inner_pos]*ztemp1
    
    Ibuy[inner_pos]=buym[inner_pos] > 0
    Isell[inner_pos]=sellm[inner_pos] > 0
    
    ncoef_b0z1[inner_pos] = (ncz>0)*ncz/dz + (ncz_zz>0)*ncz_zz/dz + (ncz_bz>0)*ncz_bz/dz 
    + PN*Ibuy[inner_pos]*czbuy[inner_pos]/dz
    
    ncoef_b01z[inner_pos] =-1*(ncz<0)*ncz/dz - (ncz_zz<0)*ncz_zz/dz - (ncz_bz<0)*ncz_bz/dz 
    - PN*Isell[inner_pos]*czsell[inner_pos]/dz
    
    ncoef_b1z0[inner_pos] = (ncb_bb>0)*ncb_bb/db + (ncb_bz>0)*ncb_bz/db 
    + PN*Ibuy[inner_pos]*(cbbuy[inner_pos]>0)*cbbuy[inner_pos]/db
    
    ncoef_1bz0[inner_pos] =-1*(ncb_bb<0)*ncb_bb/db - (ncb_bz<0)*ncb_bz/db 
    - PN*Ibuy[inner_pos]*(cbbuy[inner_pos]<0)*cbbuy[inner_pos]/db
    
    ncoef_b0z0[inner_pos] =ncu-ncoef_b0z1[inner_pos]-ncoef_b01z[inner_pos]
    -ncoef_b1z0[inner_pos]-ncoef_1bz0[inner_pos]
    
    f[inner_pos] = -1*c0[inner_pos]-nc0-nc0_zz-nc0_bb-nc0_bz
    
    #  b = blow
    ztemp0 = (u[pos_blow+nb]-u[pos_blow-nb])/dz/2
    ztemp1 = (u[pos_blow]-u[pos_blow-nb])/dz
    ztemp2 = (u[pos_blow+nb]-u[pos_blow])/dz
    btemp2 = (u[pos_blow+1]-u[pos_blow])/db
    # Non linear part from comsumption
    f_0 = exp[pos_blow]*(1-z[pos_blow]*ztemp2)
    nc0 = ko*(1/(1-gamma)*(np.abs(f_0)**a)
    -(np.abs(f_0)**(a-1))*exp[pos_blow]
    + np.abs(f_0)**a*u[pos_blow])
    
    ncz = ko*((np.abs(f_0)**(a-1))*exp[pos_blow]*z[pos_blow])
    
    ncu = -1*ko*(np.abs(f_0)**a)
    
    #Non linear part from uz^2, ub^2, ub*uz
    ncz_zz= 2*czz[pos_blow]*(1-gamma)*(czz[pos_blow]*(1-gamma)>0)*ztemp2 
    + 2*czz[pos_blow]*(1-gamma)*(czz[pos_blow]*(1-gamma)<0)*ztemp1
    
    nc0_zz = -1*czz[pos_blow]*(1-gamma)*(czz[pos_blow]*(1-gamma)>0)*ztemp2**2 
    -czz[pos_blow]*(1-gamma)*(czz[pos_blow]*(1-gamma)<0)*ztemp1**2
    
    # trading condition
    buym[pos_blow] = czbuy[pos_blow]*ztemp2 + cbbuy[pos_blow]*btemp2 
    sellm[pos_blow] = czsell[pos_blow]*ztemp1 
    Ibuy[pos_blow]=buym[pos_blow]>0
    Isell[pos_blow]=sellm[pos_blow]>0
    
    ncoef_b0z1[pos_blow] = czz[pos_blow]/dzz+cz[pos_blow]*(cz[pos_blow]>0)/dz
    + (ncz>0)*ncz/dz + (ncz_zz>0)*ncz_zz/dz + PN*Ibuy[pos_blow]*czbuy[pos_blow]/dz
    #ncoef_b0z1[pos_blow] = (ncz>0)*ncz/dz + (ncz_zz>0)*ncz_zz/dz + PN*Ibuy[pos_blow]*czbuy[pos_blow]/dz
    
    ncoef_b01z[pos_blow] =czz[pos_blow]/dzz-cz[pos_blow]*(cz[pos_blow]>0)/dz
    -1*(ncz<0)*ncz/dz - (ncz_zz<0)*ncz_zz/dz - PN*Isell[pos_blow]*czsell[pos_blow]/dz
    #ncoef_b01z[pos_blow] =-1*(ncz<0)*ncz/dz - (ncz_zz<0)*ncz_zz/dz - PN*Isell[pos_blow]*czsell[pos_blow]/dz
    
    ncoef_b1z0[pos_blow] =PN*Ibuy[pos_blow]*cbbuy[pos_blow]/db
    ncoef_b0z0[pos_blow] = ncu - ncoef_b0z1[pos_blow] - ncoef_b01z[pos_blow] - ncoef_b1z0[pos_blow]
    f[pos_blow] = -1*c0[pos_blow]-nc0-nc0_zz
    #  z = zlow, buy
    if zlow==0:
        pos=np.arange(ib1)  
        Ibuy[pos]=1
        ncoef_b1z0[pos]=PN*cbbuy[pos]/db
        ncoef_b0z0[pos]=-1*ncoef_b1z0[pos]
        pos=np.arange(ib1+1,nb)
        Ibuy[pos]=1
        ncoef_b1z0[pos]=-1*PN*cbbuy[pos]/db
        ncoef_b0z0[pos]=-1*ncoef_b1z0[pos]
        pos=ib1
        Ibuy[pos]=1
        ncoef_b1z0[pos]=PN/dz
        ncoef_b0z0[pos]=-1*ncoef_b1z0[pos]
    else:
        pos = np.arange(1,nb-1)
        Ibuy[pos]=1
        ncoef_b0z1[pos]= PN*czbuy[pos]/dz
        ncoef_b1z0[pos]= PN*(cbbuy[pos]>0)*cbbuy[pos]/db
        ncoef_1bz0[pos]=-1*PN*(cbbuy[pos]<0)*cbbuy[pos]/db
        ncoef_b0z0[pos] = -1*ncoef_b0z1[pos]-ncoef_b1z0[pos]-ncoef_1bz0[pos]
        pos = 0
        Ibuy[pos]=1
        ncoef_b0z1[pos]= PN*czbuy[pos]/dz
        ncoef_b1z0[pos]= PN*cbbuy[pos]/db
        ncoef_b0z0[pos] = -1*ncoef_b0z1[pos]- ncoef_b1z0[pos]
        pos = nb-1
        Ibuy[pos]=1
        ncoef_b0z1[pos]= PN*czbuy[pos]/dz
        ncoef_1bz0[pos]=-1*PN*cbbuy[pos]/db
        ncoef_b0z0[pos] = -1* ncoef_b0z1[pos]- ncoef_1bz0[pos]
    #   z = zup, sell
    pos = np.arange(nzb-nb,nzb)
    Isell[pos]=1
    ncoef_b01z[pos] =-1*PN*czsell[pos]/dz
    ncoef_b0z0[pos] =-1*ncoef_b01z[pos]
    #  b = bup, wash sell
    pos = np.arange(2,nz)*nb-1
    if bup>1:
        Isell[pos]=1
        ncoef_b01z[pos] =-1*PN*czsell[pos]/dz
        ncoef_b0z0[pos] =-1*ncoef_b01z[pos]
    elif bup==1:
        ncoef_b0z0[pos]=-1*PN
        f[pos]=t1*ncoef_b0z0[pos]
    p=np.arange(nzb)
    M=sparse.coo_matrix((coef_b0z0+ncoef_b0z0,(p,p)),shape=(nzb,nzb))
    p=np.arange(nzb-nb)
    M=M+sparse.coo_matrix((coef_b0z1[p]+ncoef_b0z1[p],(p,p+nb)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef_b01z[p+nb]+ncoef_b01z[p+nb],(p+nb,p)),shape=(nzb,nzb))
    p=np.arange(nzb-1)
    M=M+sparse.coo_matrix((coef_b1z0[p]+ncoef_b1z0[p],(p,p+1)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef_1bz0[p+1]+ncoef_1bz0[p+1],(p+1,p)),shape=(nzb,nzb))
    p=np.arange(nzb-nb+1)
    M=M+sparse.coo_matrix((coef_b11z[p+nb-1],(p+nb-1,p)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef_1bz1[p],(p,p+nb-1)),shape=(nzb,nzb))
    p=np.arange(nzb-nb-1)
    M=M+sparse.coo_matrix((coef_b1z1[p],(p,p+nb+1)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef_1b1z[p+nb+1],(p+nb+1,p)),shape=(nzb,nzb))
    M=M.tocsc()
    u=linalg.spsolve(M,f)
    diff = u - u_old
    num = np.linalg.norm(diff)
    deno=np.linalg.norm(u_old)
    rel =num/deno
    print('count={},error={}'.format(count,rel))
    if count%10==0:
        nontrade=(1-Isell)*(1-Ibuy)
        nontrade=nontrade.reshape((nz,nb))
        lower_bound=np.zeros((nb,))
        upper_bound=np.zeros((nb,))
        for i in range(nb):
            t=nontrade[:,i]
            index=np.where(t==1)
            if len(index[0])!=0:
                lower_bound[i]=index[0][0]*dz+zlow
                upper_bound[i]=index[0][-1]*dz+zlow
        plt.plot(b_grid,lower_bound)
        plt.plot(b_grid,upper_bound)
        plt.show()
    if rel<=Tol:
        print('count={},error={}'.format(count,rel))
        break
    elif count>=maxcountN:
        print('not converge,error={}'.format(rel))
        break
nontrade=(1-Isell)*(1-Ibuy)
nontrade=nontrade.reshape((nz,nb))
lower_bound=np.zeros((nb,))
upper_bound=np.zeros((nb,))
for i in range(nb):
    t=nontrade[:,i]
    index=np.where(t==1)
    if len(index[0])!=0:
        lower_bound[i]=index[0][0]*dz+zlow
        upper_bound[i]=index[0][-1]*dz+zlow
plt.plot(b_grid,lower_bound)
plt.plot(b_grid,upper_bound)
plt.show()
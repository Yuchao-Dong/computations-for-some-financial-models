# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:04:08 2019

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
nz = 100 
zlow=0.3
zup=0.7
nb = 110
blow = 0
bup = 1.1
PN = 1e6
Tol = 1e-9
maxcountN = 500

sgs = sigma**2
xs=(mu-r)/gamma/sigma**2
eps=math.sqrt(2*r*tau*xs/gamma/sigma**2)
kbar=beta/gamma-(1-gamma)/gamma*(r+(mu-r)**2/2/gamma/sigma**2)
ko=beta/gamma-(1-gamma)/gamma*(r*(1-tau)+(mu-r)**2/2/gamma/sigma**2)
maximum_loss=gamma/(1-gamma)*math.log(ko/kbar)

r=r*(1-tau)
t1=-6.237683

dz = (zup-zlow)/nz
db = (bup-blow)/nb
nz = nz+1
nb = nb+1
nzb = nz*nb
ib1 = round((1-blow)/db)+1
dzz = dz**2
dbb = db**2
dbz = db*dz

z_grid=np.linspace(zlow,zup,nz)
b_grid=np.linspace(blow,bup,nb)
b=matlib.repmat(b_grid,nz,1).transpose().reshape((nzb,))
z=matlib.repmat(z_grid,1,nb).reshape((nzb,))

coef1ij0 = np.zeros(nzb)
coefi1j0 =  np.zeros(nzb)
coefi0j0 =  np.zeros(nzb)
coefi0j1 =  np.zeros(nzb)
coefi01j =  np.zeros(nzb)
coef1i1j =  np.zeros(nzb)
coefi1j1 =  np.zeros(nzb)
coefi11j =  np.zeros(nzb)
coef1ij1 =  np.zeros(nzb)
f =  np.zeros(nzb)

czz = 0.5*sgs*z**2*(1-z)**2
cbz = -1*sgs*z*(1-z)*b
cbb = 0.5*sgs*b**2          
cz = ((mu-r-gamma*sgs*z)*(1-z)+r*tau/(1-tau)*b*z)*z
cb = (-1*(1-gamma)*sgs*z+sgs-mu)*b
cu = np.zeros(nzb)
c0 = (mu-r-r*tau/(1-tau)*b)*z-0.5*gamma*sgs*z**2+r-beta/(1-gamma)
a=-1*(1-gamma)/gamma
czbuy = z
cbbuy = 1-b
czsell = -1*np.ones(nzb)

pos=np.array([])
for i in range(nb-1):
    pos=np.append(pos,np.arange(i*nz+1,(i+1)*nz-1))
pos=pos.astype(int)
coefi1j0[pos] = ((cz[pos])>0)*cz[pos]/dz + czz[pos]/dzz - cbz[pos]*(cbz[pos]>0)/dbz/2 + cbz[pos]*(cbz[pos]<0)/dbz/2
coef1ij0[pos] =-1*((cz[pos])<0)*cz[pos]/dz + czz[pos]/dzz - cbz[pos]*(cbz[pos]>0)/dbz/2 + cbz[pos]*(cbz[pos]<0)/dbz/2
coefi0j1[pos] = (cb[pos]>0)*cb[pos]/db + cbb[pos]/dbb - cbz[pos]*(cbz[pos]>0)/dbz/2 + cbz[pos]*(cbz[pos]<0)/dbz/2
coefi01j[pos] =-1*(cb[pos]<0)*cb[pos]/db + cbb[pos]/dbb - cbz[pos]*(cbz[pos]>0)/dbz/2 + cbz[pos]*(cbz[pos]<0)/dbz/2
coefi1j1[pos] = cbz[pos]*(cbz[pos]>0)/dbz/2
coef1i1j[pos] = cbz[pos]*(cbz[pos]>0)/dbz/2
coefi11j[pos] =-1*cbz[pos]*(cbz[pos]<0)/dbz/2
coef1ij1[pos] =-1*cbz[pos]*(cbz[pos]<0)/dbz/2
coefi0j0[pos] = cu[pos]-coefi1j0[pos]-coef1ij0[pos]-coefi0j1[pos]-coefi01j[pos]-coefi1j1[pos]-coef1i1j[pos]-coef1ij1[pos]-coefi11j[pos]

u = np.zeros(nzb)
u = maximum_loss/2*np.ones(nzb)

countN=0

while True:
    uold=u
    countN = countN + 1 
    ncoefi0j0 = np.zeros(nzb)
    ncoefi0j1 =  np.zeros(nzb)
    ncoefi01j = np.zeros(nzb)
    ncoefi1j0 =  np.zeros(nzb)
    ncoef1ij0 =  np.zeros(nzb)
    Ibuy =  np.zeros(nzb)
    Isell =  np.zeros(nzb)
    buym =  np.zeros(nzb)
    sellm =  np.zeros(nzb)
    pos=np.array([])
    for i in range(nb-1):
        pos=np.append(pos,np.arange(i*nz+1,(i+1)*nz-1))
    pos=pos.astype(int)
    ztemp0 = (u[pos+1]-u[pos-1])/dz/2
    ztemp1 = (u[pos]-u[pos-1])/dz
    ztemp2 = (u[pos+1]-u[pos])/dz
    btemp0 = (u[pos+nz]-u[pos-nz])/db/2
    btemp1 = (u[pos]-u[pos-nz])/db
    btemp2 = (u[pos+nz]-u[pos])/db
    # Non linear part from comsumption
    f0 = np.exp(u[pos])*(1-z[pos]*ztemp2)
    nc0 = ko*(1/(1-gamma)*(np.exp(f0)**a)-(np.exp(f0)**(a-1))*np.exp(u[pos])+(np.exp(f0)**a)*u[pos])
    ncz = ko*(np.exp(f0)**(a-1)*np.exp(u[pos])*z[pos])
    ncu = -1*ko*np.exp(f0)**a
    #Non linear part from uz^2, ub^2, ub*uz
    ncz_zz= 2*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)>0)*ztemp2 + 2*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)<0)*ztemp1
    nc0_zz = -1*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)>0)*ztemp2**2 + -1*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)<0)*ztemp1**2
    ncb_bb= 2*cbb[pos]*(1-gamma)*(cbb[pos]*(1-gamma)>0)*btemp2 + 2*cbb[pos]*(1-gamma)*(cbb[pos]*(1-gamma)<0)*btemp1
    nc0_bb = -1*cbb[pos]*(1-gamma)*(cbb[pos]*(1-gamma)>0)*btemp2**2 -cbb[pos]*(1-gamma)*(cbb[pos]*(1-gamma)<0)*btemp1**2
    ncz_bz= cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)>0)*btemp2 + cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)<0)*btemp1
    ncb_bz= cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)>0)*(btemp2>0)*ztemp2 + cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)>0)*(btemp2<0)*ztemp1 + cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)<0)*(btemp1<0)*ztemp2 +cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)<0)*(btemp1<0)*ztemp1
    nc0_bz= -1*cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)>0)*btemp2*(btemp2>0)*ztemp2 -cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)>0)*btemp2*(btemp2<0)*ztemp1 -cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)<0)*btemp1*(btemp1<0)*ztemp2 -cbz[pos]*(1-gamma)*(cbz[pos]*(1-gamma)<0)*btemp1*(btemp1>0)*ztemp1
    # trading condition
    buym[pos] = czbuy[pos]*ztemp2 + (cbbuy[pos]>0)*cbbuy[pos]*btemp2+(cbbuy[pos]<0)*cbbuy[pos]*btemp1
    sellm[pos] = czsell[pos]*ztemp1
    Ibuy[pos]=buym[pos] > 0
    Isell[pos]=sellm[pos] > 0
    ncoefi1j0[pos] = (ncz>0)*ncz/dz + (ncz_zz>0)*ncz_zz/dz + (ncz_bz>0)*ncz_bz/dz + PN*Ibuy[pos]*czbuy[pos]/dz
    ncoef1ij0[pos] =-1*(ncz<0)*ncz/dz - (ncz_zz<0)*ncz_zz/dz - (ncz_bz<0)*ncz_bz/dz - PN*Isell[pos]*czsell[pos]/dz
    ncoefi0j1[pos] = (ncb_bb>0)*ncb_bb/db + (ncb_bz>0)*ncb_bz/db + PN*Ibuy[pos]*(cbbuy[pos]>0)*cbbuy[pos]/db
    ncoefi01j[pos] =-1*(ncb_bb<0)*ncb_bb/db - (ncb_bz<0)*ncb_bz/db - PN*Ibuy[pos]*(cbbuy[pos]<0)*cbbuy[pos]/db
    ncoefi0j0[pos] = ncu-ncoefi1j0[pos]-ncoef1ij0[pos]-ncoefi0j1[pos]-ncoefi01j[pos]
    f[pos] = -1*c0[pos]-nc0-nc0_zz-nc0_bb-nc0_bz
    #   b = blow
    pos =np.arange(1,nz-1)
    ztemp0 = (u[pos+1]-u[pos-1])/dz/2
    ztemp1 = (u[pos]-u[pos-1])/dz
    ztemp2 = (u[pos+1]-u[pos])/dz
    btemp2 = (u[pos+nz]-u[pos])/db
    # Non linear part from comsumption
    f0 = np.exp(u[pos])*(1-z[pos]*ztemp2)
    nc0 = ko*(1/(1-gamma)*(np.abs(f0)**a)-(np.abs(f0)**(a-1))*np.exp(u[pos])+(np.abs(f0)**a)*u[pos])
    ncz = ko*(np.abs(f0)**(a-1)*np.exp(u[pos])*z[pos])
    ncu = -1*ko*np.abs(f0)**(a)
    #Non linear part from uz^2, ub^2, ub*uz
    ncz_zz= 2*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)>0)*ztemp2 + 2*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)<0)*ztemp1;
    nc0_zz = -1*czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)>0)*ztemp2**2 -czz[pos]*(1-gamma)*(czz[pos]*(1-gamma)<0)*ztemp1**2
    #trading condition
    buym[pos] = czbuy[pos]*ztemp2 + cbbuy[pos]*btemp2 
    sellm[pos] = czsell[pos]*ztemp1 
    Ibuy[pos]=buym[pos]>0
    Isell[pos]=sellm[pos]>0
    ncoefi1j0[pos] = (ncz>0)*ncz/dz + (ncz_zz>0)*ncz_zz/dz + PN*Ibuy[pos]*czbuy[pos]/dz
    ncoef1ij0[pos] =-1*(ncz<0)*ncz/dz - (ncz_zz<0)*ncz_zz/dz - PN*Isell[pos]*czsell[pos]/dz
    ncoefi0j1[pos] = PN*Ibuy[pos]*cbbuy[pos]/db
    ncoefi0j0[pos] = ncu - ncoefi1j0[pos] - ncoef1ij0[pos] - ncoefi0j1[pos]
    f[pos] = -1*c0[pos]-nc0-nc0_zz
    if zlow==0:
        pos=np.arange(0,ib1-1)*nz 
        Ibuy[pos]=1
        ncoefi0j1[pos]=PN*cbbuy[pos]/db
        ncoefi0j0[pos]=-1*ncoefi0j1[pos]
        pos=np.arange(ib1,nb)*nz
        Ibuy[pos]=1
        ncoefi01j[pos]=-1*PN*cbbuy[pos]/db
        ncoefi0j0[pos]=-1*ncoefi01j[pos]
        pos=(ib1-1)*nz+1
        Ibuy[pos]=1
        ncoefi1j0[pos]=PN/dz
        ncoefi0j0[pos]=-1*ncoefi1j0(pos)
    else:
        pos = np.arange(1,nb-1)*nz
        Ibuy[pos]=1
        ncoefi1j0[pos]= PN*czbuy[pos]/dz
        ncoefi0j1[pos]= PN*(cbbuy[pos]>0)*cbbuy[pos]/db
        ncoefi01j[pos]=-1*PN*(cbbuy[pos]<0)*cbbuy[pos]/db
        ncoefi0j0[pos] = -1*ncoefi1j0[pos]-ncoefi0j1[pos]-ncoefi01j[pos]
        pos = 0
        Ibuy[pos]=1
        ncoefi1j0[pos]= PN*czbuy[pos]/dz
        ncoefi0j1[pos]= PN*cbbuy[pos]/db
        ncoefi0j0[pos] = -1*ncoefi1j0[pos]-ncoefi0j1[pos]
        pos = (nb-1)*nz
        Ibuy[pos]=1
        ncoefi1j0[pos]= PN*czbuy[pos]/dz
        ncoefi01j[pos]=-1*PN*cbbuy[pos]/db
        ncoefi0j0[pos] = -1*ncoefi1j0[pos]-ncoefi01j[pos]
    #   z = zup, sell
    pos = np.arange(1,nb+1)*nz-1
    Isell[pos]=1
    ncoef1ij0[pos] =-1*PN*czsell[pos]/dz
    ncoefi0j0[pos] =-1*ncoef1ij0[pos]
    pos = (nb-1)*nz+np.arange(1,nz-1)
    if bup>1:
        Isell[pos]=1
        ncoef1ij0[pos] =-1*PN*czsell[pos]/dz
        ncoefi0j0[pos] =-1*ncoef1ij0[pos]
    elif bup==1:
        ncoefi0j0[pos]=-1*PN
        f[pos]=t1*ncoefi0j0[pos]
    p=np.arange(0,nzb)
    M=sparse.coo_matrix((coefi0j0+ncoefi0j0,(p,p)),shape=(nzb,nzb))
    p=np.arange(0,nzb-1)
    M=M+sparse.coo_matrix((coefi1j0[0:nzb-1]+ncoefi1j0[0:nzb-1],(p,p+1)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef1ij0[1:]+ncoef1ij0[1:],(p+1,p)),shape=(nzb,nzb))
    p=np.arange(0,nzb-nz)
    M=M+sparse.coo_matrix((coefi0j1[0:nzb-nz]+ncoefi0j1[0:nzb-nz],(p,p+nz)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coefi01j[nz:]+ncoefi01j[nz:],(p+nz,p)),shape=(nzb,nzb))
    p=np.arange(0,nzb-nz+1)
    M=M+sparse.coo_matrix((coef1ij1[0:nzb-nz+1],(p,p+nz-1)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coefi11j[nz-1:],(p+nz-1,p)),shape=(nzb,nzb))
    p=np.arange(0,nzb-nz-1)
    M=M+sparse.coo_matrix((coefi1j1[0:nzb-nz-1],(p,p+nz+1)),shape=(nzb,nzb))
    M=M+sparse.coo_matrix((coef1i1j[1+nz:],(p+nz+1,p)),shape=(nzb,nzb))
    M=M.tocsc()
    u=linalg.spsolve(M,f)
    diff = u - uold
    num = np.linalg.norm(diff)
    deno=np.linalg.norm(uold)
    rel =num/deno
    print('count={},error={}'.format(countN,rel))
    if countN%10==0:
        nontrade=(1-Isell)*(1-Ibuy)
        nontrade=nontrade.reshape((nb,nz))
        lower_bound=np.zeros((nb,))
        upper_bound=np.zeros((nb,))
        end_point=nb-1
        for i in range(nb):
            t=nontrade[i,:]
            index=np.where(t==1)
            if len(index[0])!=0:
                lower_bound[i]=index[0][0]*dz+zlow
                upper_bound[i]=index[0][-1]*dz+zlow
            else:
                end_point=i
                break
        plt.plot(b_grid[:i],lower_bound[:i])
        plt.plot(b_grid[:i],upper_bound[:i])
        plt.show()
    if rel<=Tol:
        print('count={},error={}'.format(countN,rel))
        break
    elif countN>=maxcountN:
        print('not converge,error={}'.format(rel))
        break
   
nontrade=(1-Isell)*(1-Ibuy)
nontrade=nontrade.reshape((nb,nz))
lower_bound=np.zeros((nb,))
upper_bound=np.zeros((nb,))
end_point=nb-1
for i in range(nb):
    t=nontrade[i,:]
    index=np.where(t==1)
    if len(index[0])!=0:
        lower_bound[i]=index[0][0]*dz+zlow
        upper_bound[i]=index[0][-1]*dz+zlow
    else:
        end_point=i
        break
plt.plot(b_grid[:i],lower_bound[:i])
plt.plot(b_grid[:i],upper_bound[:i])
plt.show()
    
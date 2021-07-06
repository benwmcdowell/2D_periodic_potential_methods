from math import pi
from numpy import linspace,zeros,array,sqrt,exp
from scipy.special import mathieu_a
import matplotlib.pyplot as plt

class calculate_Mathieu_dos:
    def __init__(self,nstates,k,xpoints,ypoints,xrange,yrange,**args):
        self.energies=linspace(-xrange,xrange,xpoints) #eV
        self.A=linspace(-yrange,yrange,ypoints) #eV
        self.eigenval=zeros((xpoints,ypoints))
        self.dos=zeros((xpoints,ypoints))
        self.x=zeros((xpoints,ypoints))
        self.y=zeros((xpoints,ypoints))
        self.xrange=xrange
        self.yrange=yrange
        self.nstates=nstates
        self.k=k
        self.xpoints=xpoints
        self.ypoints=ypoints
        if 'sigma' in args:
            self.sigma=float(args['sigma'])
        else:
            self.sigma=0.001 #ev, gaussian smearing parameter
        for i in range(xpoints):
            for j in range(ypoints):
                self.x[i][j]=self.energies[i]
                self.y[i][j]=self.A[j]
        self.h=6.626e-34 #J s
        self.h/=2*pi
        self.m=9.10938356e-31 #kg
        if 'me' in args:
            self.me=args['me']
        else:
            self.me=1.0
        self.m*=self.me
        self.b=1.6022e-19 #J/eV
        
    def main(self):
        if type(self.k)=='list':
            for i in self.k:
                self.calculate_dos[k]
        else:
            self.calculate_dos[self.k]
        self.plot_dos()
        
    def calculate_dos(self,k):
        for n in range(1,self.nstates):
            for j in range(self.ypoints):
                q=self.A[j]*self.b*self.m/pi**2/k**2/self.h**2
                if n%2==0:
                    a=mathieu_a(n,q)
                    a*=pi**2*k**2*h**2/m/b/2
                    a-=-xrange
                    a=round(a/(2*self.xrange/self.xpoints))
                    if a>0 and a<xpoints:
                        self.eigenval[a][j]+=1.0
        for j in range(self.ypoints):
            smeared_dos=zeros(self.xpoints)
            for i in range(self.xpoints):
                gauss=array([(self.eigenval[i][j]/self.sigma/sqrt(2*pi))*exp((((i-j)*2*self.xrange/self.xpoints)/self.sigma)**2/-2) for k in range(self.xpoints)])
                smeared_dos+=gauss
            self.dos[:,j]+=smeared_dos
    
    def plot_dos(self):
        plt.figure()
        plt.title('Mathieu equation density of states')
        plt.pcolormesh(self.x,self.y,self.dos,cmap='jet',shading='nearest')
        plt.xlabel('energy / eV')
        plt.ylabel('barrier height / eV')
        cbar=plt.colorbar()
        cbar.set_label('density of states / states $eV^{-1}$')
        plt.show()
        
        plt.figure()
        plt.title('Mathieu eigenenergies')
        plt.pcolormesh(self.x,self.y,self.eigenval,cmap='jet',shading='nearest')
        plt.xlabel('energy / eV')
        plt.ylabel('barrier height / eV')
        cbar=plt.colorbar()
        cbar.set_label('number of states')
        plt.show()

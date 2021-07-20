from math import pi
from numpy import linspace,zeros,array,sqrt,exp,floor
from numpy.linalg import norm
from scipy.special import mathieu_a
import matplotlib.pyplot as plt
from json import load
from sys import exit

class calculate_Mathieu_dos:
    def __init__(self,data_type,xpoints,ypoints,xrange,yrange,**args):
        self.data_type=data_type
        self.eigenval=zeros((ypoints,xpoints))
        self.dos=zeros((ypoints,xpoints))
        self.psi=zeros((ypoints,xpoints))
        self.psi_smeared=zeros((ypoints,xpoints))
        self.x=zeros((ypoints,xpoints))
        self.y=zeros((ypoints,xpoints))
        self.xpoints=xpoints
        self.ypoints=ypoints
        if 'sigma' in args:
            self.sigma=float(args['sigma'])
        else:
            self.sigma=0.001 #ev, gaussian smearing parameter
        if 'auto_axes' in args:
            self.auto_axes=True
        else:
            self.auto_axes=False
        self.h=6.626e-34 #J s
        self.h/=2*pi
        self.m=9.10938356e-31 #kg
        if 'me' in args:
            self.me=args['me']
        else:
            self.me=0.4
        self.m*=self.me
        self.b=1.6022e-19 #J/eV
        self.k=0.0
        
        if self.data_type!='energy' and self.data_type!='function':
            print('unknown data type. use either function, for plotting real space projections of Mathieu DOS, or energy, for plotting Mathieu DOS as a function of potential barrier height')
            exit()
            
        if type(yrange)==list:
            tempy=linspace(min(yrange),max(yrange),ypoints)
            self.yrange=max(yrange)-min(yrange)
        else:
            tempy=linspace(-yrange/2,yrange/2,ypoints)
            self.yrange=yrange
        if type(xrange)==list:
            tempx=linspace(min(xrange),max(xrange),xpoints)
            self.xrange=max(xrange)-min(xrange)
        else:
            tempx=linspace(-xrange/2,xrange/2,xpoints)
            self.xrange=xrange
            
        for i in range(self.ypoints):
            for j in range(self.xpoints):
                self.x[i][j]=tempx[j]
                self.y[i][j]=tempy[i] 
        
    def read_json_eigenenergies(self,k,filepath,**args):
        self.k=k
        if 'normalize_dos' in args:
            normalize=True
        else:
            normalize=False
        with open(filepath) as file:
            data=load(file)
            data=array([[float(i[j]) for j in range(1,len(i))] for i in data[1:]])
            data*=pi**2*self.k**2*self.h**2/self.m/self.b/2
            data+=abs(self.x)
        if self.auto_axes:
            tol=max(data.flatten())-min(data.flatten())
            tol*=0.02
            self.yrange=[min(data.flatten())-tol,max(data.flatten())+tol]
            tempy=linspace(min(self.yrange),max(self.yrange),self.ypoints)
            for i in range(self.ypoints):
                for j in range(self.xpoints):
                    self.y[i][j]=tempy[i]
            self.yrange=self.yrange[1]-self.yrange[0]
        for i in range(self.xpoints):
            for a in data[:,i]:
                a-=self.y[0][0]
                a=round(a/(self.yrange/self.ypoints))
                if a>0 and a<self.ypoints:
                    self.eigenval[a][i]+=1.0
        if self.sigma!=0.0:
            for j in range(self.xpoints):
                smeared_dos=zeros(self.ypoints)
                for i in range(self.ypoints):
                    if normalize:
                        gauss=array([(self.eigenval[i][j]/self.sigma/sqrt(2*pi))*exp((((i-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)])  #normalized gaussian
                    if not normalize:
                        gauss=array([self.eigenval[i][j]*exp((((i-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)]) #unnormalized gaussian
                    smeared_dos+=gauss
                self.dos[:,j]+=smeared_dos
        self.data_type='energy'
        
    def read_json_eigenfunctions(self,filepath,**args):
        if 'normalize_dos' in args:
            normalize=True
        else:
            normalize=False
        if 'reduced_zone' in args:
            reduced=True
            periods=int(args['reduced_zone'])
            if self.xpoints%periods!=0:
                print('indivisible number of periods selected')
                exit()
        else:
            reduced=False
        with open(filepath) as file:
            data=load(file)
        for i in range(1,len(data)):
            for j in range(1,len(data[i])):
                if type(data[i][j])==list:
                    self.psi[i-1][j-1]=0.0
                else:
                    self.psi[i-1][j-1]=float(data[i][j])**2
        for i in range(self.ypoints):
            if max(self.psi[i])>0.0:
                self.psi[i]/=norm(self.psi[i])
        if self.sigma!=0.0:
            for i in range(self.xpoints):
                smeared_dos=zeros(self.ypoints)
                for j in range(self.ypoints):
                    if normalize:
                        gauss=array([(self.psi[j][i]/self.sigma/sqrt(2*pi))*exp((((j-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)])  #normalized gaussian
                    if not normalize:
                        gauss=array([self.psi[j][i]*exp((((j-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)]) #unnormalized gaussian
                    smeared_dos+=gauss
                self.psi_smeared[:,i]+=smeared_dos
        if reduced:
            self.x=self.x[:,:int(1/periods*self.xpoints)]
            self.x-=self.x[0][0]
            self.y=self.y[:,:int(1/periods*self.ypoints)]
            new_psi=zeros((self.ypoints,int(self.xpoints/periods)))
            new_psi_smeared=zeros((self.ypoints,int(self.xpoints/periods)))
            for i in range(periods):
                new_psi+=self.psi[:,int(i/periods*self.xpoints):int((i+1)/periods*self.xpoints)]
                new_psi_smeared+=self.psi[:,int(i/periods*self.xpoints):int((i+1)/periods*self.xpoints)]
            self.psi=new_psi
            self.psi_smeared=new_psi_smeared
        self.data_type='function'
        
    def sum_2D(self,filepaths,**args):
        if self.data_type=='energy':
            if 'k' not in args:
                print('supply the spatial frequencies (in units of per m) as a list: ie k=[k1,k2]')
                exit()
            k=args['k']
            for i in range(len(filepaths)):
                tempvar=calculate_Mathieu_dos(self.data_type,self.xpoints,self.ypoints,[self.x[0][0],self.x[0][-1]],[self.y[0][0],self.y[-1][0]],me=self.me,sigma=self.sigma)
                tempvar.read_json_eigenenergies(k[i],filepaths[i])
                self.eigenval+=tempvar.eigenval
                self.dos+=tempvar.dos
        if self.data_type=='function':
            if 'reduced_zone' in args:
                reduced=True
                periods=int(args['reduced_zone'])
                self.x=self.x[:,:int(1/periods*self.xpoints)]
                self.x-=self.x[0][0]
                self.y=self.y[:,:int(1/periods*self.ypoints)]
                new_psi=zeros((self.ypoints,int(self.xpoints/periods)))
                new_psi_smeared=zeros((self.ypoints,int(self.xpoints/periods)))
                for i in range(periods):
                    new_psi+=self.psi[:,int(i/periods*self.xpoints):int((i+1)/periods*self.xpoints)]
                    new_psi_smeared+=self.psi[:,int(i/periods*self.xpoints):int((i+1)/periods*self.xpoints)]
                self.psi=new_psi
                self.psi_smeared=new_psi_smeared
            else:
                reduced=False
            for i in range(len(filepaths)):
                tempvar=calculate_Mathieu_dos(self.data_type,self.xpoints,self.ypoints,[self.x[0][0],self.x[0][-1]],[self.y[0][0],self.y[-1][0]],me=self.me,sigma=self.sigma)
                if reduced:
                    tempvar.read_json_eigenfunctions(filepaths[i],reduced_zone=periods)
                else:
                    tempvar.read_json_eigenfunctions(filepaths[i])
                self.psi+=tempvar.psi
                self.psi_smeared+=tempvar.psi_smeared
        
    def calculate_dos(self,k,nstates):
        self.nstates=nstates
        for n in range(1,self.nstates):
            for j in range(self.ypoints):
                q=self.A[j]*self.b*self.m/pi**2/k**2/self.h**2
                if n%2==0:
                    a=mathieu_a(n,q)
                    a*=pi**2*k**2*self.h**2/self.m/self.b/2
                    a-=-self.energies[0]
                    a=round(a/(self.xrange/self.xpoints))
                    if a>0 and a<self.xpoints:
                        self.eigenval[a][j]+=1.0
        for j in range(self.ypoints):
            smeared_dos=zeros(self.xpoints)
            for i in range(self.xpoints):
                gauss=array([(self.eigenval[i][j]/self.sigma/sqrt(2*pi))*exp((((i-k)*2*self.xrange/self.xpoints)/self.sigma)**2/-2) for k in range(self.xpoints)])
                smeared_dos+=gauss
            self.dos[:,j]+=smeared_dos
    
    def plot_dos(self,n,a,**args):
        if 'title' in args:
            title=str(args['title'])
        else:
            title=''
        if self.data_type=='energy':
            if self.sigma!=0.0:
                plt.figure()
                if len(title)==0:
                    plt.title('Mathieu density of states | $\sigma$ = {}'.format(self.sigma))
                else:
                    plt.title(title+' | $\sigma$ = {}'.format(self.sigma))
                plt.pcolormesh(self.x,self.y,self.dos,cmap='jet',shading='nearest')
                plt.ylabel('relative energy / eV')
                plt.xlabel('barrier height / eV')
                cbar=plt.colorbar()
                cbar.set_label('density of states / states $eV^{-1}$')
                plt.show()
            
            plt.figure()
            if len(title)==0:
                plt.title('Mathieu eigenenergies | $\sigma$ = 0.0')
            else:
                plt.title(title+' | $\sigma$ = 0.0')
            plt.pcolormesh(self.x,self.y,self.eigenval,cmap='jet',shading='nearest')
            plt.ylabel('relative energy / eV')
            plt.xlabel('barrier height / eV')
            cbar=plt.colorbar()
            cbar.set_label('number of states')
            plt.show()
            
        if self.data_type=='function':     
            if self.sigma!=0.0:
                plt.figure()
                if len(title)==0:
                    plt.title('Mathieu density of states | $\sigma$ = {}'.format(self.sigma))
                else:
                    plt.title(title+' | $\sigma$ = {}'.format(self.sigma))
                for i in range(-n,n+1):
                    plt.pcolormesh(self.x+i*a,self.y,self.psi_smeared,cmap='jet',shading='nearest')
                    if i<n:
                        plt.plot([i*a+a/2 for j in range(2)],[self.y[0][0],self.y[-1][0]],linestyle='dashed',color='white')
                plt.ylabel('relative energy / eV')
                plt.xlabel('position / $\AA^{-1}$')
                cbar=plt.colorbar()
                cbar.set_label('density of states / states $eV^{-1}$')
                plt.show()
            
            plt.figure()
            if len(title)==0:
                plt.title('Mathieu functions | $\sigma$ = 0.0')
            else:
                plt.title(title+' | $\sigma$ = 0.0')
            for i in range(-n,n+1):
                plt.pcolormesh(self.x+i*a,self.y,self.psi,cmap='jet',shading='nearest')
                if i<n:
                    plt.plot([i*a+a/2 for j in range(2)],[self.y[0][0],self.y[-1][0]],linestyle='dashed',color='white')
            plt.ylabel('relative energy / eV')
            plt.xlabel('position / $\AA^{-1}$')
            cbar=plt.colorbar()
            cbar.set_label('density of states / states $eV^{-1}$')
            plt.show()
from math import pi
import numpy as np
from numpy import linspace,zeros,array,sqrt,exp,ceil,argmin,shape,abs,average,cos,diag
from numpy.linalg import norm
from scipy.special import mathieu_a
import matplotlib.pyplot as plt
from json import load
from sys import exit
from time import time
from copy import deepcopy
from scipy.fft import fft,fftfreq
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter
import os
from scipy.optimize import curve_fit

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
            self.sigma=0.0 #ev, gaussian smearing parameter
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
            self.me=1.0
        self.m*=self.me
        self.b=1.6022e-19 #J/eV
        self.sigmax=0.0
        
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
            
        if 'k' in args:
            self.k=args['k']
        else:
            self.k=self.xrange
            
        for i in range(self.ypoints):
            for j in range(self.xpoints):
                self.x[i][j]=tempx[j]
                self.y[i][j]=tempy[i] 
        
        self.start=time()
        
    def read_json_eigenenergies(self,k,filepath,**args):
        self.k=k
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
        self.data_type='energy'
        
    def smear_energies(self,sigma,**args):
        if 'normalize' in args:
            normalize=args['normalize']
        else:
            normalize=True
        self.sigma=sigma
        scale=sqrt(2*pi)*self.sigma
        if self.data_type=='energy':
            percentage_counter=[25,75,100]
            for j in range(self.xpoints):
                smeared_dos=zeros(self.ypoints)
                for i in range(self.ypoints):
                    if normalize:
                        gauss=array([(self.eigenval[i][j]/self.sigma/sqrt(2*pi))*exp((((i-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)])  #normalized gaussian
                    if not normalize:
                        gauss=array([self.eigenval[i][j]*exp((((i-k)*self.yrange/self.ypoints)/self.sigma)**2/-2) for k in range(self.ypoints)]) #unnormalized gaussian
                    smeared_dos+=gauss
                self.dos[:,j]+=smeared_dos
                if round(j/(self.xpoints-1)*100)%25==0 and round(j/(self.xpoints-1)) in percentage_counter:
                    print('{}% finished with Gaussian energy smearing routine. {} s elasped so far'.format(round(j/(self.xpoints-1)*100),time()-self.start))
                    try:
                        percentage_counter.remove(round(j/(self.xpoints-1)*100))
                    except ValueError:
                        pass
        elif self.data_type=='function':
            psi_smeared_copy=deepcopy(self.psi_smeared)
            percentage_counter=[25,50,75]
            mask=array([exp(((j*self.yrange/self.ypoints)/self.sigma)**2/-2) for j in range(self.ypoints)])
            for i in range(self.xpoints):
                for j in range(self.ypoints):
                    if normalize:
                        gauss=array([psi_smeared_copy[j][i]/scale*mask[abs(j-k)] for k in range(self.ypoints)])  #normalized gaussian
                    elif not normalize:
                        gauss=array([psi_smeared_copy[j][i]*mask[abs(j-k)] for k in range(self.ypoints)]) #unnormalized gaussian
                    self.psi_smeared[:,i]+=gauss
                if round(i/(self.xpoints-1)*100)%25==0 and round(i/(self.xpoints-1)*100) in percentage_counter:
                    print('{}% finished with Gaussian energy smearing routine. {} s elapsed so far'.format(round(i/(self.xpoints-1)*100),time()-self.start))
                    try:
                        percentage_counter.remove(round(i/(self.xpoints-1)*100))
                    except ValueError:
                        pass
        
    def smear_spatial(self,sigmax,**args):
        if 'normalize' in args:
            normalize=args['normalize']
        else:
            normalize=True
        self.sigmax=sigmax
        psi_smeared_copy=deepcopy(self.psi_smeared)
        percentage_counter=[25,50,75]
        x0=int(ceil(5*self.sigmax*self.xpoints/self.xrange))
        mask=array([exp((j*self.xrange/self.xpoints/self.sigmax)**2/-2) for j in range(self.xpoints+2*x0)])
        scale=sqrt(2*pi)*self.sigmax
        for i in range(self.ypoints):
            for j in range(-x0,self.xpoints+x0):
                if normalize:
                    gauss=array([psi_smeared_copy[i][j%self.xpoints]/scale*mask[abs(j-k)] for k in range(self.xpoints)])  #normalized gaussian
                if not normalize:
                    gauss=array([psi_smeared_copy[i][j%self.xpoints]*mask[abs(j-k)] for k in range(self.xpoints)]) #unnormalized gaussian
                self.psi_smeared[i]+=gauss
            if round(i/(self.ypoints-1)*100)%25==0 and round(i/(self.ypoints-1)*100) in percentage_counter:
                print('{}% finished with Gaussian spatial smearing routine. {} s elapsed so far'.format(round(i/(self.ypoints-1)*100),time()-self.start))
                try:
                    percentage_counter.remove(round(i/(self.ypoints-1)*100))
                except ValueError:
                    pass
                
    def read_json_eigenfunctions(self,filepath,**args):
        self.eigenval=zeros(self.ypoints)
        self.energies=[]
        self.momenta=[]
        os.chdir(filepath)
        if 'derivative' in args and args['derivative']==True:
            derivative=True
            file_header='derivatives'
        else:
            derivative=False
            file_header='functions'
        if 'parity' in args:
            parity=args['parity']
        else:
            parity=['odd','even']
        if 'sigmax' in args:
            self.sigmax=float(args['sigmax'])
        if 'prob_density' in args:
            prob_density=args['prob_density']
        else:
            prob_density=True
        if 'offset' in args:
            self.eoffset=args['offset']
        else:
            self.eoffset=0.0
        if 'reduced_zone' in args:
            reduced=True
            periods=int(args['reduced_zone'])
            if self.xpoints%periods!=0:
                print('indivisible number of periods selected')
                exit()
        else:
            reduced=False
        for p in parity:
            with open('Mathieu_'+file_header+'_'+p+'.json') as file:
                data=load(file)
            with open('Mathieu_energies_'+p+'.json') as file:
                edata=load(file)
            with open('Mathieu_k_'+p+'.json') as file:
                kdata=load(file)
            for i in range(1,len(data)):
                if type(edata[i])==list:
                    pass
                else:
                    counter=round((self.ypoints-1)*(float(edata[i])+self.eoffset-min(self.y[:,0]))/self.yrange)
                    self.energies.append(edata[i])
                    self.momenta.append(kdata[i])
                    if counter>0 and counter<self.ypoints:
                        self.eigenval[counter]+=1
                        for j in range(1,len(data[i])):
                            if type(data[i][j])==list:
                                pass
                            else:
                                if prob_density:
                                    self.psi[counter][j-1]+=float(data[i][j])**2
                                else:
                                    self.psi[counter][j-1]+=float(data[i][j])
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
            
        self.energies=array(self.energies)
        self.momenta=array(self.momenta)/self.k*pi
        self.momenta=self.momenta[self.energies.argsort()]
        self.energies=self.energies[self.energies.argsort()]
        self.psi_smeared=deepcopy(self.psi)
        self.data_type='function'
        
    def normalize(self):
        for i in range(self.ypoints):
            if max(self.psi[i])>0.0:
                self.psi[i,:]*=self.eigenval[i]/norm(self.psi[i,:])
                
    def make_supercell(self,num):
        self.x+=self.xrange/2
        self.xpoints=round(self.xpoints*num)
        self.xrange*=num
        new_psi=zeros((self.ypoints,self.xpoints))
        new_psi_smeared=zeros((self.ypoints,self.xpoints))
        newx=zeros((self.ypoints,self.xpoints))
        newy=zeros((self.ypoints,self.xpoints))
        for i in range(num):
            new_psi[:,i*self.xpoints//num:(i+1)*self.xpoints//num]+=self.psi
            new_psi_smeared[:,i*self.xpoints//num:(i+1)*self.xpoints//num]+=self.psi_smeared
            newy[:,i*self.xpoints//num:(i+1)*self.xpoints//num]+=self.y
            newx[:,i*self.xpoints//num:(i+1)*self.xpoints//num]+=self.x+self.xrange/num*(i-num/2)
        self.psi=new_psi
        self.psi_smeared=new_psi_smeared
        self.x=newx
        self.y=newy
        
    def add_scattering_source(self,spoint,smag):
        self.scattered=zeros((self.ypoints,self.xpoints))
        #spoint is the horizontal index at which the scattering point sits
        Eweight=array([1/(1+(self.h**2*i/self.b*self.yrange/(self.ypoints-1)*2/smag**2/self.m*1e20)) for i in range(self.ypoints)])
        for i in range(self.xpoints):
            r=abs(i-spoint)*self.xrange/(self.xpoints-1)
            self.psi[:,i]+=self.psi[:,i]/sqrt(r)*Eweight
            self.psi_smeared[:,i]+=self.psi_smeared[:,i]/sqrt(r)*Eweight
            self.scattered[:,i]+=self.psi[:,i]/sqrt(r)*Eweight
        
    def sum_2d(self,fp,x2npts=0,y2npts=0,k2=157.86353/2,y2range=0,de=0.05,tol=0,offset=0.0,normalize_weights=True,normalize_pos=True):
        if x2npts==0:
            x2npts=self.xpoints
        if y2npts==0:
            y2npts=self.ypoints
        if y2range==0:
            y2range=[self.y[0,0],self.y[0,0]+self.yrange]
        if tol==0:
            tol=self.y[1,0]-self.y[0,0]            
        
        def calc_weighting(e1,e2,esum,de):
            return np.exp(-abs(e1+e2-esum)/de)
        psi_2d=np.zeros((self.ypoints,self.xpoints,x2npts))
        
        self.other_psi=calculate_Mathieu_dos('function',x2npts,y2npts,k2,y2range)
        self.other_psi.read_json_eigenfunctions(fp)
        
        #test data
        #self.psi=np.zeros(np.shape(self.psi))
        #for i in range(self.ypoints):
        #    self.psi[i,:]+=np.array([j for j in range(self.xpoints)])
        #self.other_psi.psi=np.zeros(np.shape(self.other_psi.psi))
        #for i in range(y2npts):
        #    self.other_psi.psi[i,:]+=np.array([j for j in range(x2npts)])
        
        if normalize_weights:
            self.psi/=np.max(self.psi)
            self.other_psi.psi/=np.max(self.other_psi.psi)
        
        for j in range(self.ypoints):
            for k in range(y2npts):
                i=np.argmin(abs(self.y[:,0]-self.y[j,0]-self.other_psi.y[k,0]))
                if abs(self.y[i,0]-self.y[j,0]-self.other_psi.y[k,0])<tol:
                    tempx,tempy=np.meshgrid(self.psi[j,:],self.other_psi.psi[k,:])
                    psi_2d[i,:,:]+=(tempx*tempy).T
                

                
        #for i in range(self.ypoints):
            #for k in range(x2npts):
                #psi_2d[i,:,k]+=self.psi[i,:]
                
            #j=np.argmin(abs(self.y[i,0]-self.other_psi.y[:,0]))
            #for k in range(self.xpoints):
            #    psi_2d[i,k,:]+=self.other_psi.psi[j,:]
                    
        self.psi_2d=psi_2d
        
        if normalize_pos:
            for i in range(self.xpoints):
                for j in range(x2npts):
                    self.psi_2d[:,i,j]/=sum(self.psi_2d[:,i,j])
        
        if de!=0:
            #gaussian smearing
            de/=(self.y[1,0]-self.y[0,0])
            for i in range(self.xpoints):
                for j in range(x2npts):
                    self.psi_2d[:,i,j]=gaussian_filter(self.psi_2d[:,i,j],de,mode='nearest')
                
        self.y+=offset
    
    def plot_sum_2d(self,pos,axis=0,cmap='vivid'):
        fig_2dsum,ax_2dsum=plt.subplots(1,1,tight_layout=True)
        if axis==0:
            ax_2dsum.pcolormesh(self.x,self.y,self.psi_2d[:,:,pos]/np.max(self.psi_2d[:,:,pos]),shading='nearest',cmap=cmap)
        elif axis==1:
            ax_2dsum.pcolormesh(self.other_psi.x,self.y,self.psi_2d[:,pos,:]/np.max(self.psi_2d[:,pos,:]),shading='nearest',cmap=cmap)
        ax_2dsum.set(xlabel='position / $\AA$', ylabel='energy / eV')
        fig_2dsum.show()
        
    def plot_slice_sum_2d(self,pos,pos2,axis=0):
        fig_sumslice,ax_sumslice=plt.subplots(1,1,tight_layout=True)
        if type(pos)!=list:
            pos=[pos]
        if type(pos2)!=list:
            pos2=[pos2]
        if axis==0:
            for i,j in zip(pos,pos2):
                ax_sumslice.plot(self.y[:,i],self.psi_2d[:,i,j])
        if axis==1:
            for i,j in zip(pos,pos2):
                ax_sumslice.plot(self.y[:,i],self.psi_2d[:,j,i])
        ax_sumslice.set(xlabel='energy / eV', ylabel='LDOS / a.u.')
        fig_sumslice.show()
            
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
            
    def overlay_tunneling_probability(self,d,phi,normalize=True):
        def tunneling_factor(V,E,phi):
            V*=1.60218e-19
            E*=1.60218e-19
            phi*=1.60218e-19
            prefactor=8/3/V*pi*sqrt(2*9.11e-31)/6.626e-34
            barrier=(phi-E+V)**(3/2)-(phi-E)**(3/2)
            
            return prefactor*barrier
        
        for i in range(self.ypoints):
            k=tunneling_factor(abs(self.y[i][0]),abs(self.y[i][0]),phi)
            counter=0
            while np.isnan(k):
                k=tunneling_factor(abs(self.y[i+counter][0]),abs(self.y[i+counter][0]),phi)
                counter+=1
            self.psi[i]*=exp(-k*1e-10*d)
            self.psi_smeared[i]*=exp(-k*1e-10*d)
            
        self.psi/=np.max(self.psi)
        self.psi_smeared/=np.max(self.psi_smeared)

    def overlay_potential(self,A):
        plt.plot(self.x[0,:],A*cos(self.x[0,:]*2*pi/self.xrange)+self.eoffset+A,color='white',linestyle='dashed')
        plt.show()
        
    def find_bandgap(self):
        max_diff=0
        for i in range(len(self.energies)-1):
            if self.energies[i+1]-self.energies[i]>max_diff:
                max_diff=self.energies[i+1]-self.energies[i]
                oband=self.energies[i]
        print('band gap: {} eV'.format(max_diff))
        print('highest occupied band: {} eV below Fermi level'.format(oband))
            
    def plot_dispersion(self,relative_energies=False,**args):
        def parabola_fit(x,a,b):
            y=a*x**2+b
            return y
        
        if 'fit' in args:
            fit=args['fit']
        else:
            fit=True
            
        if 'erange' in args:
            erange=args['erange']
        else:
            erange=(min(self.energies),max(self.energies))
        erange=tuple([argmin(abs(self.energies-erange[i])) for i in range(2)])
        
        if relative_energies:
            self.energies-=min(self.energies)
        
        efit=[]
        kfit=[]
        counter=erange[0]
        while len(efit)<3:
            if self.momenta[counter] not in kfit or self.energies[counter] not in efit:
                efit.append(self.energies[counter])
                kfit.append(self.momenta[counter])
            counter+=1
        efit=array(efit)
        kfit=array(kfit)
            
        if 'overlay' in args:
            overlay_fit=True
            m=args['overlay'][0]
            m_err=args['overlay'][1]
        else:
            overlay_fit=False
            
        plt.figure()
        plt.scatter(self.momenta[erange[0]:erange[1]],self.energies[erange[0]:erange[1]],label='raw data')
        if fit:
            p0=((max(efit)-min(efit))/(max(kfit**2)-min(kfit**2)),-(max(efit)-min(efit))/(max(kfit**2)-min(kfit**2))*kfit[0]**2+efit[0])
            bounds=((0,-100),(100,100))
            popt,pcov=curve_fit(parabola_fit,kfit*1e10,efit*self.b,p0=p0)
            plt.plot(self.momenta[erange[0]:erange[1]],parabola_fit(self.momenta[erange[0]:erange[1]]*1e10,popt[0],popt[1])/self.b,label='fit')
            me=self.h**2/2/popt[0]/self.m
            pcov=sqrt(diag(pcov))
            print('m* = {} +/- {}'.format(me,pcov[0]/popt[0]*me))
        if overlay_fit:
            A=self.h**2/2/m/self.m
            A_err=m_err/m*A
            plt.errorbar(self.momenta[erange[0]:erange[1]],parabola_fit(self.momenta[erange[0]:erange[1]]*1e10,A,0.0)/self.b+0.3,yerr=(self.momenta[erange[0]:erange[1]])**2*A_err,fmt='o',label='fit')
        plt.xlabel('momentum / radians $\AA^{-1}$')
        plt.ylabel('energy / eV')
        plt.tight_layout()
        plt.legend()
        plt.show()
            
    def plot_fft(self,**args):
        plt.figure()
        if 'window' in args:
            w=hann(self.xpoints,sym=False)
        else:
            w=array([1.0 for i in range(self.xpoints)])
        if 'normalize' in args:
            normalize=True
        else:
            normalize=False
            
        zf=zeros((self.ypoints,self.xpoints//2))
        xf=zeros((self.ypoints,self.xpoints//2))
        for i in range(self.ypoints):
            zf[i]+=abs(fft((self.psi_smeared[i]-average(self.psi_smeared[i]))*w)[0:self.xpoints//2])*2.0/self.xpoints
            if normalize:
                zf[i]/=sum(zf[i])
            xf[i]+=fftfreq(self.xpoints,self.xrange/(self.xpoints-1))[:self.xpoints//2]
        plt.pcolormesh(xf,self.y[:,:self.xpoints//2],zf,cmap='jet',shading='nearest')
        plt.xlabel('momentum / $\AA^{-1}$')
        plt.ylabel('energy / eV')
        plt.tight_layout()
        plt.show()
        
    def plot_position_slice(self,pos):
        plt.figure()
        for p in pos:
            i=argmin(abs(self.x[0,:]-p))
            plt.plot(self.y[:,i],self.psi_smeared[:,i],label=p)
        plt.xlabel('energy - $E_{F}$ / eV')
        plt.ylabel('LDOS')
        plt.legend()
        plt.show()
    
    def plot_dos(self,n,a,**args):
        if 'overlay_potential' in args:
            overlay_potential=True
            potential=args['overlay_potential']
        else:
            overlay_potential=False
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        if 'title' in args:
            self.title=str(args['title'])
        else:
            self.title=''
        if self.data_type=='energy':
            if self.sigma!=0.0:
                plt.figure()
                if len(self.title)==0:
                    title='Mathieu density of states\n$\sigma_{energy}$ = ' + str(self.sigma)
                else:
                    title=self.title+'\n$\sigma_{energy}$ = '+str(self.sigma)+' eV'
                plt.title(title)
                plt.pcolormesh(self.x,self.y,self.dos,cmap=cmap,shading='nearest')
                plt.ylabel('relative energy / eV')
                plt.xlabel('barrier height / eV')
                cbar=plt.colorbar()
                cbar.set_label('density of states / states $eV^{-1}$')
                plt.tight_layout()
                plt.show()
            
            plt.figure()
            if len(self.title)==0:
                title='Mathieu eigenenergies\n$\sigma_{energy}$ = 0.0 eV'
            else:
                title=self.title+'\n$\sigma_{energy}$ = 0.0 eV'
            plt.title(title)
            plt.pcolormesh(self.x,self.y,self.eigenval,cmap=cmap,shading='nearest')
            plt.ylabel('relative energy / eV')
            plt.xlabel('barrier height / eV')
            cbar=plt.colorbar()
            cbar.set_label('number of states')
            plt.tight_layout()
            plt.show()
            
        if self.data_type=='function':  
            plt.figure()
            if len(self.title)==0:
                title='Mathieu functions\n$\sigma_{energy}$ = 0.0 eV | $\sigma_{spatial}$ = 0.0 $\AA$'
            else:
                title=self.title+'\n$\sigma_{energy}$ = 0.0 eV | $\sigma_{spatial}$ = 0.0 $\AA$'
            plt.title(title)
            for i in range(-n,n+1):
                plt.pcolormesh(self.x+i*a,self.y,self.psi,cmap=cmap,shading='nearest')
                if overlay_potential:
                    plt.plot(self.x[0,:]+i*a,potential*cos(self.x[0,:]*2*pi/self.xrange)+self.eoffset+potential,color='white',linestyle='dashed')
                if i<n:
                    plt.plot([i*a+a/2 for j in range(2)],[self.y[0][0],self.y[-1][0]],linestyle='dashed',color='white')
            plt.ylabel('relative energy / eV')
            plt.xlabel('position / $\AA$')
            cbar=plt.colorbar()
            cbar.set_label('density of states / states $eV^{-1}$')
            plt.tight_layout()
            plt.show()
        
            if self.sigma!=0.0 or self.sigmax!=0.0:
                plt.figure()
                if len(self.title)==0:
                    title='Mathieu density of states\n$\sigma_{energy}$ = '+str(self.sigma)+' eV'
                else:
                    title=self.title+'\n$\sigma_{energy}$ = '+str(self.sigma)+' eV'
                if self.sigmax!=0.0:
                    title+=' | $\sigma_{spatial}$ = '+str(self.sigmax)+' $\AA$'             
                plt.title(title)
                for i in range(-n,n+1):
                    plt.pcolormesh(self.x+i*a,self.y,self.psi_smeared,cmap=cmap,shading='nearest')
                    if overlay_potential:
                        plt.plot(self.x[0,:]+i*a,potential*cos(self.x[0,:]*2*pi/self.xrange)+self.eoffset+potential,color='white',linestyle='dashed')
                    if i<n:
                        plt.plot([i*a+a/2 for j in range(2)],[self.y[0][0],self.y[-1][0]],linestyle='dashed',color='white')
                plt.ylabel('relative energy / eV')
                plt.xlabel('position / $\AA$')
                cbar=plt.colorbar()
                cbar.set_label('density of states / states $eV^{-1}$')
                plt.tight_layout()
                plt.show()
                
        print('total elapsed time: {} s'.format(time()-self.start))
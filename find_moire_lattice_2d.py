import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool
import getopt
import json
import sys
import os

class moire():
    def __init__(self):
        pass
    
    def write_file(self,ofile):
        with open(ofile+'_differences','w+') as f:
            f.write(json.dumps([list(i) for i in [self.mindiff]]))
        with open(ofile+'_coefficients','w+') as f:
            f.write(json.dumps([list(self.coeff[:,i]) for i in range(4)]))
            
    def read_file(self,ifile):
        with open(ifile+'_diff','r') as f:
            self.mindiff=json.load(f)[0]
        with open(ifile+'_coefficients','r') as f:
            self.coeff=json.load(f)
        self.apts=len(self.mindiff)
        self.angle=np.array([i*np.pi*2/(self.apts-1) for i in range(self.apts)])
            
    def calculate(self,nprocs,apts,slv,alv,**args):
        if 'max_length' in args:
            self.max_length=args['max_length']
        else:
            self.max_length=100
        print('max vector length is: {}'.format(self.max_length))
        self.alv=alv
        self.slv=slv
        self.apts=apts
        self.nprocs=nprocs
        self.angle=np.array([i*np.pi*2/(apts-1) for i in range(apts)])
        self.mindiff=np.zeros((2,self.apts))
        self.coeff=np.zeros((2,self.apts,4))
        if self.nprocs>1:
            pool=ProcessPool(self.nprocs)
            output=pool.map(self.single_process, [i for i in range(self.apts)])
            for i in range(self.apts):
                self.mindiff+=output[i][0]
                self.coeff+=output[i][1]
            pool.close()
        else:
            for i in range(self.apts):
                tempvar1,tempvar2=self.single_process(i)
                for j in range(2):
                    self.mindiff[j,i]=tempvar1[j,i]
                    self.coeff[j,i]+=tempvar2[j,i]
        
    def single_process(self,a):
        import numpy as np
        lpts=np.array([int(np.round(self.max_length/np.linalg.norm(self.alv[i]))) for i in range(2)])
        lrange=[[j for j in range(-lpts[i],lpts[i]+1)] for i in range(2)]
        for i in range(2):
            lrange[i].remove(0)
        tempdiff=np.zeros((2,self.apts))
        coeff=np.zeros((2,self.apts,4))
        rot=np.array([[np.cos(self.angle[a]),np.sin(self.angle[a])],[-np.sin(self.angle[a]),np.cos(self.angle[a])]])
        for k in range(2):
            tempdiff[k,a]=np.inf
            for i in lrange[0]:
                for j in lrange[1]:
                    temp_coeff=np.array([0,0,i,j])
                    pos=self.alv[0]*i+self.alv[1]*j
                    pos=np.dot(pos,rot)
                    pos=np.dot(pos,np.linalg.inv(self.slv))
                    while pos[k]>1.0 or pos[k]<0.0:
                        if pos[k]>1.0:
                            pos[k]-=1.0
                            temp_coeff[k]+=1
                        if pos[k]<0.0:
                            pos[k]+=1.0
                            temp_coeff[k]-=1
                    pos=np.dot(pos,self.slv)
                    if np.linalg.norm(pos)<tempdiff[k,a]:
                        tempdiff[k,a]=np.linalg.norm(pos)
                        coeff[k,a]=temp_coeff
                    
        return tempdiff,coeff
    
    def plot_moire(self):
        self.angle*=180/np.pi
        self.fig,self.axs=plt.subplots(3,1,sharex=True)
        for i in range(3):
            if i<2:
                self.axs[i].plot(self.angle,self.mindiff[i])
            else:
                self.axs[i].plot(self.angle,self.mindiff[0]+self.mindiff[1])
            self.axs[i].set(ylabel='lattice mismatch / nm')
        self.axs[-1].set(xlabel='misorientation angle / degrees')
        self.fig.show()
    
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    nprocs=1
    output='./moire_output'
    try:
        opts,args=getopt.getopt(sys.argv[1:],'p:s:a:n:o:m:',['processors=','substrate=','adlayer=','npts=','output=','max_length='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-p', '--processors']:
            nprocs=int(j)
        if i in ['-s','--substrate']:
            slv=np.array([float(k) for k in j.split(',')]).reshape((2,2))
        if i in ['-a','--adlayer']:
            alv=np.array([float(k) for k in j.split(',')]).reshape((2,2))
        if i in ['-n','--npts']:
            apts=int(j)
        if i in ['-o','--output']:
            output=j
        if i in ['-m','--max_length']:
            max_length=float(j)
        else:
            max_length=np.inf
    try:
        main=moire()
        main.calculate(nprocs,apts,slv,alv,max_length=max_length)
        main.write_file(output)
    except NameError:
        print('Error: lattice vectors not defined')
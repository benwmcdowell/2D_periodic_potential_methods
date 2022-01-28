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
        with open(ofile,'w+') as f:
            f.write(json.dumps([list(i) for i in self.mindiff]))
            
    def read_file(self,ifile):
        with open(ifile,'r') as f:
            self.mindiff=json.load(f)
        self.apts=np.shape(self.mindiff)[0]
        self.angle=np.array([i*np.pi*2/(self.apts-1) for i in range(self.apts)])
            
    def calculate(self,nprocs,apts,lpts,slv,alv,elv,**args):
        if 'max_length' in args:
            self.max_length=args['max_length']
        else:
            self.max_length=np.inf
        self.alv=alv
        self.slv=slv
        self.elv=elv
        self.apts=apts
        self.lpts=lpts
        self.nprocs=nprocs
        self.angle=np.array([i*np.pi*2/(apts-1) for i in range(apts)])
        pool=ProcessPool(self.nprocs)
        output=pool.map(self.single_process, [i for i in range(self.apts)])
        self.mindiff=sum(output)
        pool.close()
        
    def single_process(self,a):
        import numpy as np
        lrange=[int(i) for i in range(-self.lpts,self.lpts+1)]
        lrange.remove(0)
        tempdiff=np.zeros((2,self.apts))
        for i in range(2):
            tempdiff[i,a]=np.inf
        rot=np.array([[np.cos(self.angle[a]),np.sin(self.angle[a])],[-np.sin(self.angle[a]),np.cos(self.angle[a])]])
        for i in lrange:
            for j in lrange:
                pos=self.slv[0]*i+self.slv[1]*j
                tempalv=np.array([np.dot(k,rot) for k in self.alv])
                pos=np.dot(pos,np.linalg.inv(tempalv))
                if np.linalg.norm(pos)<self.max_length:
                    for k in range(2):
                        while pos[k]>1.0 or pos[k]<0.0:
                            if pos[k]>1.0:
                                pos[k]-=1.0
                            if pos[k]<0.0:
                                pos[k]+=1.0
                    pos=np.dot(pos,tempalv)
                    if np.linalg.norm(pos)<tempdiff[0,a]:
                        tempdiff[0,a]=np.linalg.norm(pos)
                        pos=self.slv[0]*i+self.slv[1]*j
                        for k in lrange:
                            for l in lrange:
                                for m in lrange:
                                    epos=elv[0]*k+elv[1]*l+elv[2]*m
                                    if np.linalg.norm(pos-epos)<tempdiff[1,a]:
                                        tempdiff[1,a]=np.linalg.norm(pos-epos)
        return tempdiff
    
    def plot_moire(self):
        self.angle*=180/np.pi
        self.fig,self.axs=plt.subplots(2,1,sharex=True)
        for i in range(2):
            self.axs[i].plot(self.angle,self.mindiff[i])
        self.axs[-1].set(ylabel='minimum difference / nm')
        self.axs.set(xlabel='misorientation angle / degrees')
        self.fig.show()
    
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    nprocs=1
    output='./moire_output'
    try:
        opts,args=getopt.getopt(sys.argv[1:],'p:s:a:n:e:o:m:',['processors=','substrate=','adlayer=','npts=','exp=','output=','max_length='])
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
        if i in ['-e','--exp']:
            elv=np.array([float(k) for k in j.split(',')]).reshape((len(j.split(','))//2,2))
        if i in ['-n','--npts']:
            apts,lpts=j.split(',')
            apts=int(apts)
            lpts=int(lpts)
        if i in ['-o','--output']:
            output=j
        if i in ['-m','--max_length']:
            max_length=float(j)
        else:
            max_length=np.inf
    try:
        main=moire()
        main.calculate(nprocs,apts,lpts,slv,alv,elv,max_length=max_length)
        main.write_file(output)
    except NameError:
        print('Error: lattice vectors not defined')
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
        self.apts=np.shape(self.mindiff)[1]
        self.angle=np.array([i*np.pi*2/(self.apts-1) for i in range(self.apts)])
            
    def calculate(self,nprocs,apts,lpts,slv,alv,elv):
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
        output=pool.map(self.single_process, [i for i in range(self.apts) for j in range(np.shape(elv)[0])], [j for i in range(self.apts) for j in range(np.shape(elv)[0])])
        self.mindiff=sum(output)
        pool.close()
        
    def single_process(self,a,e):
        import numpy as np
        lrange=[int(i) for i in range(-self.lpts,self.lpts+1)]
        lrange.remove(0)
        tempdiff=np.zeros((np.shape(self.elv)[0],self.apts))
        tempdiff[e,a]+=np.inf
        rot=np.array([[np.cos(self.angle[a]),np.sin(self.angle[a])],[-np.sin(self.angle[a]),np.cos(self.angle[a])]])
        for i in lrange:
            for j in lrange:
                for k in lrange:
                    for l in lrange:
                        pos=self.slv[0]*i+self.slv[1]*j+np.dot(self.alv[0]*k,rot)+np.dot(self.alv[1]*l,rot)
                        if pos<self.max_length:
                            pos=pos%self.elv[e]
                            if np.linalg.norm(pos)<tempdiff[e,a]:
                                tempdiff[e,a]=np.linalg.norm(pos)
        return tempdiff
    
    def plot_moire(self):
        self.angle*=180/np.pi
        self.fig,self.axs=plt.subplots(np.shape(self.mindiff)[0]+1,1,sharex=True)
        for i in range(np.shape(self.mindiff)[0]):
            self.axs[i].plot(self.angle,self.mindiff[i])
            self.axs[i].set(ylabel='minimum difference / $\AA$')
        self.axs[-1].plot(self.angle,sum(np.array([self.mindiff[i] for i in range(np.shape(self.mindiff)[0])])))
        self.axs[-1].set(xlabel='misorientation angle / degrees')
        self.fig.show()
    
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    nprocs=1
    output='./moire_output'
    try:
        opts,args=getopt.getopt(sys.argv[1:],'p:s:a:n:e:o:',['processors=','substrate=','adlayer=','npts=','exp=','-output'])
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
    try:
        main=moire()
        main.calculate(nprocs,apts,lpts,slv,alv,elv)
        main.write_file(output)
    except NameError:
        print('Error: lattice vectors not defined')
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool
import getopt
import json

class moire():
    def __init__(self):
        pass
    
    def output(self,ofile):
        with open(ofile,'w') as f:
            f.write(json.dump(self.mindiff))
            
    def read_file(self,ifile):
        with open(ifile,'r') as f:
            self.mindiff=json.load(f)
            
    def calculate(self,nprocs,apts,lpts,slv,alv,elv):
        self.alv=alv
        self.slv=slv
        self.apts=apts
        self.lpts=lpts
        self.nprocs=nprocs
        pool=ProcessPool(self.nprocs)
        output=pool.map(self.single_process, [i for i in range(self.apts) for j in range(np.shape(elv)[0])], [j for i in range(self.apts) for j in range(np.shape(elv)[0])])
        self.mindiff=sum(output)
        pool.close()
        
    def single_process(self,angle,e):
        import numpy as np
        lrange=[int(i) for i in range(-self.lpts,self.lpts+1)]
        lrange.remove(0)
        tempdiff=np.zeros(np.shape(self.elv)[0],self.apts)
        tempdiff[angle,e]+=1000000
        angle*=np.pi*2/(self.apts-1)
        rot=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        for i in lrange:
            for j in lrange:
                for k in lrange:
                    for l in lrange:
                        pos=self.slv[0]*i+self.slv[1]*j+np.linalg.dot(self.alv[0]*k,rot)+np.linalg.dot(self.alv[1]*l,rot)
                        pos=pos%self.elv[e]
                        if np.linalg.norm(pos)<tempdiff:
                            tempdiff=np.linalg.norm(pos)
        return tempdiff
    
if __name__ == '__main__':
    pass
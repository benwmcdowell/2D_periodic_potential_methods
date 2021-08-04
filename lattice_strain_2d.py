from numpy import array,dot,sqrt,zeros,argmin,argmax,shape
from numpy.linalg import inv,norm
from shutil import copyfile,rmtree
import matplotlib.pyplot as plt
import os.path
import os

def create_VASP_directories(template,output,npts,lvmin,lvmax):
    if os.path.exists(output):
        os.chdir(output)
    else:
        os.mkdir(output)
        os.chdir(output)
    for i in range(npts):
        for j in range(npts):
            sf=[(lvmax-lvmin)*k/(npts-1)+lvmin for k in [i,j]]
            name=str(sf[0])+','+str(sf[1])
            if os.path.exists(name):
                rmtree(name)
            os.mkdir(name)
            
            for k in ['KPOINTS','INCAR','POTCAR']:
                copyfile(os.path.join(template,k),os.path.join(output,name,k))
            with open(os.path.join(template,'job.sh')) as file:
                lines=file.readlines()
            for k in range(len(lines)):
                if 'job-name' in lines[k]:
                    tempvar=lines[k].split('=')
                    lines[k]=tempvar[0]+'='+name+' '+tempvar[1]
            with open(os.path.join(output,name,'job.sh'),'w') as file:
                for k in lines:
                    file.write(k)
            
            lv,coord,atomtypes,atomnums=parse_poscar(os.path.join(template,'POSCAR'))
            for k in range(len(coord)):
                coord[k]=dot(coord[k],inv(lv))
            for k in range(2):
                lv[k]*=sf[k]
            for k in range(len(coord)):
                coord[k]=dot(coord[k],lv)
            write_poscar(os.path.join(output,name,'POSCAR'),lv,coord,atomtypes,atomnums)
    
    print(str(npts**2)+' VASP directories written')
    
def plot_band_gap(filepath,lv_ref,**args):
    if 'tol' in args:
        tol=args['tol']
    else:
        tol=10
    files=os.listdir(filepath)
    npts=int(sqrt(len(files)))
    unsortedx=[]
    unsortedy=[]
    unsortedz=[]
    lv_ref=parse_poscar(lv_ref)[0][:2]
    lv_ref=[norm(i) for i in lv_ref]
    for i in range(npts**2):
        os.chdir(filepath)
        os.chdir(os.path.join(filepath,files[i]))
        try:
            dos,energies=parse_doscar('./DOSCAR')[:2]
            total_dos=zeros(len(energies))
            for j in range(1,len(dos)):
                for k in dos[j]:
                    total_dos+=k
            unsortedz.append(find_band_gap(energies,total_dos,tol=tol))
        except:
            print('error in directory: {}'.format(files[i]))
            unsortedz.append(0.0)
            pass
        lv=parse_poscar('./POSCAR')[0]
        unsortedx.append((norm(lv[0])/lv_ref[0]-1)*100)
        unsortedy.append((norm(lv[1])/lv_ref[1]-1)*100)
        
    x=zeros((npts,npts))
    y=zeros((npts,npts))
    z=zeros((npts,npts))
    for i in range(npts):
        for j in range(npts):
            tempvar=[[],[],[]]
            for k in range(len(unsortedy)):
                if unsortedy[k]==min(unsortedy):
                    tempvar[0].append(unsortedx[k])
                    tempvar[1].append(unsortedy[k])
                    tempvar[2].append(unsortedz[k])
            k=argmin(tempvar[0])
            x[i][j]=tempvar[0][k]
            y[i][j]=tempvar[1][k]
            z[i][j]=tempvar[2][k]
            unsortedx.remove(tempvar[0][k])
            unsortedy.remove(tempvar[1][k])
            unsortedz.remove(tempvar[2][k])
            
    plt.figure()
    plt.pcolormesh(x,y,z,shading='nearest',cmap='jet')
    plt.xlabel('% distortion of lattice vector #1')
    plt.ylabel('% distortion of lattice vector #2')
    cbar=plt.colorbar()
    cbar.set_label('band gap / eV')
    plt.tight_layout()
    plt.show()
            
def find_band_gap(energies,dos,**args):
    if 'tol' in args:
        tol=args['tol']
    else:
        tol=10
    edge_energies=[0.0,0.0]
    for i in range(tol,len(energies)-tol):
        if max(dos[i-tol:i+tol])==dos[i]:
            if energies[i]<0.0:
                edge_energies[0]=energies[i]
            if energies[i]>0.0:
                edge_energies[1]=energies[i]
                break
    
    bg=edge_energies[1]-edge_energies[0]
    return bg

#reads DOSCAR
def parse_doscar(filepath):
    with open(filepath,'r') as file:
        line=file.readline().split()
        atomnum=int(line[0])
        for i in range(5):
            line=file.readline().split()
        nedos=int(line[2])
        ef=float(line[3])
        dos=[]
        energies=[]
        for i in range(atomnum+1):
            if i!=0:
                line=file.readline()
            for j in range(nedos):
                line=file.readline().split()
                if i==0:
                    energies.append(float(line[0]))
                if j==0:
                    temp_dos=[[] for k in range(len(line)-1)]
                for k in range(len(line)-1):
                    temp_dos[k].append(float(line[k+1]))
            dos.append(temp_dos)
    energies=array(energies)-ef
    
    #orbitals contains the type of orbital found in each array of the site projected dos
    num_columns=shape(dos[1:])[1]
    if num_columns==3:
        orbitals=['s','p','d']
    elif num_columns==6:
        orbitals=['s_up','s_down','p_up','p_down','d_up','d_down']
    elif num_columns==9:
        orbitals=['s','p_y','p_z','p_x','d_xy','d_yz','d_z2','d_xz','d_x2-y2']
    elif num_columns==18:
        orbitals=['s_up','s_down','p_y_up','p_y_down','p_z_up','p_z_down','p_x_up','p_x_down','d_xy_up','d_xy_down','d_yz_up','d_yz_down','d_z2_up','d_z2_down','d_xz_up','d_xz_down','d_x2-y2_up','d_x2-y2_down']
        
    #dos is formatted as [[total dos],[atomic_projected_dos for i in range(atomnum)]]
    #total dos has a shape of (4,nedos): [[spin up],[spin down],[integrated, spin up],[integrated spin down]]
    #atomic ldos have shapes of (6,nedos): [[i,j] for j in [spin up, spin down] for i in [s,p,d]]
    #energies has shape (1,nedos) and contains the energies that each dos should be plotted against
    return dos, energies, ef, orbitals

#reads POSCAR
def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=dot(coord[i],latticevectors)
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums

#writes POSCAR
def write_poscar(ofile, lv, coord, atomtypes, atomnums, **args):
    with open(ofile,'w') as file:
        if 'title' in args:
            file.write(str(args['title']))
        file.write('\n1.0\n')
        for i in range(3):
            for j in range(3):
                file.write(str('{:<018f}'.format(lv[i][j])))
                if j<2:
                    file.write('  ')
            file.write('\n')
        for i in atomtypes:
            file.write('  '+str(i))
        file.write('\n')
        for i in atomnums:
            file.write('  '+str(i))
        file.write('\n')
        if 'seldyn' in args:
            file.write('Selective Dynamics\n')
        file.write('Direct\n')
        for i in range(len(coord)):
            coord[i]=dot(coord[i],inv(lv))
        for i in range(len(coord)):
            for j in range(3):
                file.write(str('{:<018f}'.format(coord[i][j])))
                if j<2:
                    file.write('  ')
            if 'seldyn' in args:
                for j in range(3):
                    file.write('  ')
                    file.write(args['seldyn'][i][j])
            file.write('\n')

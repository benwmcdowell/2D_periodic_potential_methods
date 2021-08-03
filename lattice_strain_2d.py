from numpy import array,dot
from shutil import copyfile
from os import mkdir,chdir
from os.path import exists
from sys import exit

def create_VASP_directories(template,output,npts,lvmin,lvmax):
    if exists(output):
        chdir(output)
    else:
        mkdir(output)
        chdir(output)
    for i in range(npts):
        for j in range(npts):
            sf=[(lvmax-lvmin)*k/(npts-1)+lvmin for k in [i,j]]
            name=str(sf[0])+','+str(sf[1])
            mkdir(name)
            
            for k in ['KPOINTS','INCAR','POTCAR']:
                copyfile(template+k,output+name+k)
            with open(template+'job.sh') as file:
                lines=file.readlines()
                for k in range(len(lines)):
                    if 'job-name' in lines[j]:
                        tempvar=lines[k].split('=')
                        lines[k]=tempvar[0]+'='+name+' '+tempvar[1]
            with open(output+name+'job.sh','w') as file:
                for k in lines:
                    file.write(k)
            
            lv,coord,atomtypes,atomnums=parse_poscar(template+'POSCAR')
            for k in range(2):
                lv[k]*=sf[k]
            write_poscar(output+name,lv,coord,atomtypes,atomnums)
    
    print(str(npts**2)+' VASP directories written')

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
    print('new POSCAR written to: '+str(ofile))
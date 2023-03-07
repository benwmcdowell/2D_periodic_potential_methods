import matplotlib.pyplot as plt
import numpy as np

#elv is the experimental lattice vectors to be matched
#lv1 and lv2 are the theoretical lattice vectors defining the moire pattern, which form an incident angle of theta
def plot_moire_angle_real(elv,lv1,name1,lv2,name2,**args):
    if 'npts' in args:
        npts=args['npts']
    else:
        npts=361
    theta=np.array([i/(npts-1)*360 for i in range(npts)])
    theta*=np.pi/180
    
    if 'errors' in args:
        errors=args['errors']
    else:
        errors=None
    
    if 'reciprocal' in args:
        for i in range(2):
            elv[i]/=(np.linalg.norm(elv[i]))**2
            elv[i]*=np.pi*2
        
            tempvar=np.zeros((3,3))
            tempvar[:2,:2]+=lv1
            tempvar[2,2]=1.0
            lv1[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
            lv1[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
            
            tempvar=np.zeros((3,3))
            tempvar[:2,:2]+=lv2
            tempvar[2,2]=1.0
            lv2[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
            lv2[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
            
    if errors:
        for i in range(2):
            errors[i]*=np.linalg.norm(elv[i])
            
    if 'rotate_lv2' in args:
        theta2=30/180*np.pi
        rot=np.array([[np.cos(theta2),np.sin(theta2)],[-np.sin(theta2),np.cos(theta2)]])
        for i in range(2):
            lv2[i]=np.dot(lv2[i],rot)
            tempvar=np.linalg.norm(lv2[i])
            lv2[i]/=tempvar
            lv2[i]*=np.cos(theta2)*tempvar/2
    
    diff=np.zeros((2,npts))
    shifts=np.zeros((2,npts,4))
    for i in range(2):
        for j in range(npts):
            rot=np.array([[np.cos(theta[j]),np.sin(theta[j])],[-np.sin(theta[j]),np.cos(theta[j])]])
            pos=np.dot(elv[i],np.linalg.inv(lv1))
            for k in range(2):
                while pos[k]>1.0 or pos[k]<0.0:
                    if pos[k]>1.0:
                        pos[k]-=1.0
                        shifts[i,j,k]+=1
                    elif pos[k]<0.0:
                        pos[k]+=1.0
                        shifts[i,j,k]-=1
            pos=np.dot(pos,lv1)
            
            pos=np.dot(pos,np.linalg.inv(np.dot(lv2,rot)))
            for k in range(2):
                while pos[k]>1.0 or pos[k]<0.0:
                    if pos[k]>1.0:
                        pos[k]-=1.0
                        shifts[i,j,k+2]+=1
                    if pos[k]<0.0:
                        pos[k]+=1.0
                        shifts[i,j,k+2]-=1
            pos=np.dot(pos,np.dot(lv2,rot))
            
            diff[i,j]+=np.linalg.norm(pos)
            
    theta*=180/np.pi
                        
    fig,axs=plt.subplots(3,1)
    for i in range(2):
            axs[i].plot(theta,diff[i])
            if errors:
                axs[i].plot(theta,diff[i]-errors[i],label='min')
                axs[i].plot(theta,diff[i]+errors[i],label='max')
    axs[2].plot(theta,diff[0]+diff[1])
    
    for i in range(3):
        axs[i].plot(theta,np.zeros(npts),linestyle='dashed',color='black')
    
    if errors:
        axs[2].plot(theta,diff[0]+diff[1]-np.sqrt(errors[0]**2+errors[1]**2),label='min')
        axs[2].plot(theta,diff[0]+diff[1]+np.sqrt(errors[0]**2+errors[1]**2),label='max')
            
    axs[-1].set(xlabel='theta / degrees')
    axs[0].set(ylabel='$V_{e1}$')
    axs[1].set(ylabel='$V_{e2}$')
    axs[2].set(ylabel='$V_{e1} + V_{e2}$')
    if errors:
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
    fig.show()
    
    print('errors:')
    print(errors)
    
    return shifts,theta
    
    
def plot_moire_angle_rec(elv,lv1,name1,lv2,name2,**args):
    if 'errors' in args:
        errors=args['errors']
    else:
        errors=None
        
    npts=3601
    theta=np.array([i/(npts-1)*360 for i in range(npts)])
    theta*=np.pi/180
    
    for i in range(2):
        elv[i]/=(np.linalg.norm(elv[i]))**2
        elv[i]*=np.pi*2
        
    if errors:
        for i in range(2):
            errors[i]*=np.linalg.norm(elv[i])
    
    tempvar=np.zeros((3,3))
    tempvar[:2,:2]+=lv1
    tempvar[2,2]=1.0
    lv1[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    lv1[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    
    tempvar=np.zeros((3,3))
    tempvar[:2,:2]+=lv2
    tempvar[2,2]=1.0
    lv2[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    lv2[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    
    if 'rotate_lv2' in args:
        theta2=30/180*np.pi
        rot=np.array([[np.cos(theta2),np.sin(theta2)],[-np.sin(theta2),np.cos(theta2)]])
        for i in range(2):
            lv2[i]=np.dot(lv2[i],rot)
            tempvar=np.linalg.norm(lv2[i])
            lv2[i]/=tempvar
            lv2[i]*=np.cos(theta2)*tempvar/2
            
    if 'reduce_elv' in args:
        for i in range(2):
            elv[i]=np.dot(elv[i],np.linalg.inv(lv1))
            for j in range(2):
                while elv[i][j]>1.0 or elv[i][j]<0.0:
                    if elv[i][j]>1.0:
                        elv[i][j]-=1.0
                    elif elv[i][j]<0.0:
                        elv[i][j]+=1.0
            elv[i]=np.dot(elv[i],lv1)
        
    print(elv)
    print(lv1)
    print(lv2)
    
    diff=np.zeros((2,4,npts))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(npts):
                    rot=np.array([[np.cos(theta[l]),np.sin(theta[l])],[-np.sin(theta[l]),np.cos(theta[l])]])
                    
                    pos=[elv[i]*(-2*j+1)+(-2*k+1)*lv1[m] for m in range(2)]                 
                        
                    diff[i,2*j+k,l]=min(np.linalg.norm(m+np.dot(lv2[0],rot)-np.dot(lv2[1],rot)) for m in pos)
                        
    theta*=180/np.pi
    
    minval=np.zeros((4,2))
    minangle=np.zeros((4,2))
    for i in range(2):
        for j in range(4):
            minval[j,i]=np.min(diff[i,j])
            minangle[j,i]=theta[np.argmin(diff[i,j])]
            
    print('minimum differences:')
    print(minval)
    print('angles of minimum differences:')
    print(minangle)
    print('errors:')
    print(errors)
    
    fig,axs=plt.subplots(4,2,sharex=True)
    for i in range(2):
        for j in range(4):
            axs[j,i].plot(theta,diff[i,j])
            if errors:
                axs[j,i].plot(theta,diff[i,j]+errors[i],label='max')
                axs[j,i].plot(theta,diff[i,j]-errors[i],label='min')
        axs[-1,i].set(xlabel='theta / degrees')
        
    axs[0,0].set(title='Ve1')
    axs[0,1].set(title='Ve2')
    axs[0,0].set(ylabel='Va + Ve')
    axs[1,0].set(ylabel='-Va + Ve')
    axs[2,0].set(ylabel='+Va - Ve')
    axs[3,0].set(ylabel='-Va - Ve')
    if errors:
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
    fig.show()
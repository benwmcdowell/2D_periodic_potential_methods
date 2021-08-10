import matplotlib.pyplot as plt


def plot_Moire_2d(a,b,**args):
    if 'tolerance' in args:
        tol=args['tolerance']
    else:
        tol=[i*0.01 for i in a]
        
    if 'distort' in args:
        distort=True
    else:
        distort=False
    
    x=[[],[]]
    y=[[],[]]
    for i in range(2):
        dx=a[i]
        counter=0
        searching=True
        while searching:
            x[i].append(counter*a[i])
            dx=min([abs(counter*a[i]-j*b[i]) for j in range(0,2*round(a[i]/b[i])*(counter+1))])
            y[i].append(dx)
            if dx<tol[i] and counter>1:
                searching=False
                if distort:
                    if (counter*a[i]+dx)%b[i]>(counter*a[i]-dx)%b[i]:
                        dx*=-1
                    for j in range(len(x[i])):
                        x[i][j]-=dx*(j/(len(x[i])-1))
                        y[i][j]-=dx*(j/(len(y[i])-1))
                    print('lattice vector #{} distorted by {} %'.format(i,dx/x[i][-1]*100))
            counter+=1
                    
    fig,ax=plt.subplots(1,1)
    for i in range(2):
        ax.scatter(x[i],y[i],s=500)
    ax.set_xlabel('distance along lattice vector / $\AA$')
    ax.set_ylabel('distance between nearest lattice sites / $\AA$')
    plt.tight_layout()
    plt.show()
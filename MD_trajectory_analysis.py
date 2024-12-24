import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

eles=['C','H','CHs','C2']
fig, axs = plt.subplots(ncols=6,nrows=4, sharey='row', figsize=(16, 16),constrained_layout=True)

for ind,ele in enumerate(eles):
    neb=np.loadtxt(ele+'-neb-coordN')
    data=np.loadtxt(ele+'-800-coordN')
    x=data[:,0]
    y=data[:,1]
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    xgrid = np.linspace(xmin, xmax, 100)
    ygrid = np.linspace(ymin, ymax, 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    T=[600,800,1000,1200,1400,1600]

    for i in range(6):
        filename=ele+'-'+str(T[i])+"-coordN"

        data=np.loadtxt(filename)
        if(ele=='C' or ele=='H' or ele=='CHs'):
            box=[[xmin,ymin],[xmax,ymax]]
        else:
            box=[[xmin,ymin,0,0],[xmax,ymax,0,0]]
        data=np.append(data,box,axis=0)
        x=data[7000:8000,0]
        y=data[7000:8000,1]
        kde=gaussian_kde(np.vstack([x,y]))
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        np.savetxt("Z-"+ele+"-"+str(T[i]),Z)

        ax = axs[ind,i]
        #hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='viridis')
        ax.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[xmin,xmax,ymin,ymax],
           cmap='viridis')        
        #ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        if(ind==0):
            ax.set_title(str(T[i])+" K",fontsize=16,weight="bold")
        if(i==0):
            ax.set_ylabel("Weighted Height ($\AA$)",fontsize=16)

        if(ind==3):
            ax.set_xlabel("CN",fontsize=16)
        #cb = fig.colorbar(hb, ax=ax)
        #ax.plot(neb[:,0],neb[:,1],'w-')
        ax.plot(neb[:,0],neb[:,1],'w-',linewidth=2,linestyle='dotted')
        ax.plot(neb[0,0],neb[0,1],'o',markersize=4,color='chocolate')
        ax.plot(neb[8,0],neb[8,1],'ro',markersize=4)
        ax.tick_params(axis='both', which='major', labelsize=15)
        #cb.set_label('log10(N)')

#plt.show()
plt.savefig('gaussian_kde_only_1ps.jpg',dpi=300)

import numpy as np 

target=open("erf_terrain_def","w")

x=np.arange(512,1800,16)
y=np.arange(512,1800,16)

X,Y=np.meshgrid(x,y)

for i in range(0,X.shape[0]):
    for j in range(0,X.shape[1]):
        target.write("%g %g %g\n"%(X[i,j],Y[i,j],32))
target.close()

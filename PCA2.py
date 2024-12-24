from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from dscribe.kernels import REMatchKernel
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import sklearn.preprocessing
import os

frames=ase.io.read("../all.xyz", index=slice(None))
unique=np.unique(frames[0].get_atomic_numbers())
#force_mean=np.ndarray(len(frames))
#force_mean.fill(0)
idx=0
for frame in frames:
  unique=np.unique(np.concatenate((unique, frame.get_atomic_numbers())))
#  force_mean[idx]=np.abs(frame.get_forces()).mean()
  frame.wrap()
  idx+=1

print("Number of frames in total: ", len(frames))
print("Atomic numbers: ", unique.tolist())
#print("Average forces of each frame: ", force_mean)
nframes = len(frames)
#idx = np.random.choice(np.arange(nframes), 1000, replace=False)
#frames_less = [frames[i] for i in idx.tolist()]
#force_mean_less = force_mean[idx]
frames_less=frames
#force_mean_less = force_mean

print("Building kernels...")
#计算描述符，此处选择SOAP
desc = SOAP(species=unique.tolist(), rcut=5.0, nmax=8, lmax=6, sigma=0.2, periodic=True, crossover=True, sparse=False, average=True)
frames_features_less = []
"""
把一个体系所有原子的描述符拼起来构成一个描述符矩阵
需要设定average（上面）和normalize（下面）
"""
for frame in frames_less:
  frames_features_less.append(desc.create(frame, n_jobs=8,positions=[80]))

#re = AverageKernel(metric="linear")
#re_kernel = re.create(frames_features_less)
print("Kernel PCA...")
for frames_features in frames_features_less:
  frames_features = sklearn.preprocessing.normalize(frames_features)
#re = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
#re_kernel = re.create(frames_features_less)

frames_features_less_np = np.concatenate(frames_features_less)
#frames_features_less_np = sklearn.preprocessing.normalize(frames_features_less_np)
#print("Kernel PCA...")
#print(re_kernel)

#kpca = KernelPCA(kernel="precomputed", fit_inverse_transform=False, gamma=10, n_components = 2, n_jobs=4)
#kpca = KernelPCA(kernel="linear", fit_inverse_transform=False, gamma=10, n_components = 2, n_jobs=4)
"""
可以看成如下过程：
对体系的描述符矩阵进行降维，然后对角化得到特征值和特征向量
选最大的两个特征值对应的特征向量作为x，y轴
"""
kpca = KernelPCA(kernel="linear", fit_inverse_transform=False, gamma=10, n_components = 4, n_jobs=8)
X_kpca = kpca.fit_transform(frames_features_less_np)
#X_kpca = kpca.fit_transform(re_kernel)
#X[ind]=X_kpca

"""
根据atomic force的大小，给点赋予不同的颜色
"""
#一般不应该超过正负0.05，不然就是很不合理的构型了
#plt.xlim(-0.03,0.05)
#plt.ylim(-0.02,0.04)

fnames=['3000 K','2000 K','1800 K','1600 K','1400 K','1200 K','1000 K','800 K','600 K','AIMD']
framenums=np.array([50, 50, 200,    400,    500,     400,      250,    100,     50,   2280])

colors=['red','orange','orchid','steelblue','fuchsia','cyan','blueviolet','olivedrab','dimgrey','green','darkcyan','purple','goldenrod','hotpink','tomato']

plt.figure(figsize=(9,5))
for ind,lab in enumerate(fnames):
  if ind==0:
    ind_low=0
  else:
    ind_low=framenums[:ind].sum()
  ind_high=framenums[:ind+1].sum()
  print(ind_low,ind_high)
  plt.scatter(X_kpca[ind_low:ind_high, 2], X_kpca[ind_low:ind_high, 3], c=colors[ind], s=2, alpha=1, edgecolor=colors[ind],label=lab)
  
plt.legend(fontsize=15)
plt.xlabel("Principal component 1",fontsize=15)
plt.ylabel("Principal component 2",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("Csoap.jpg",ppi=300)
plt.clf()

plt.figure(figsize=(9,5))
ind_low=framenums[:-1].sum()
ind_high=framenums.sum()
print(ind_low,ind_high)

plt.scatter(X_kpca[:ind_low, 2], X_kpca[:ind_low, 3], c=colors[0], s=2, alpha=1, edgecolor=colors[0],label="GAP")
plt.scatter(X_kpca[ind_low:ind_high, 2], X_kpca[ind_low:ind_high, 3], c=colors[9], s=2, alpha=1, edgecolor=colors[9],label="AIMD")

plt.legend(fontsize=15)
plt.xlabel("Principal component 1",fontsize=15)
plt.ylabel("Principal component 2",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("Csoap2.jpg",ppi=300)

os.system("convert -trim Csoap.jpg Csoap.jpg")
os.system("convert -trim Csoap2.jpg Csoap2.jpg")

from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from sklearn.cluster import k_means

from scipy.spatial.distance import cdist

def computeHistogram(img_file, F, textons):
    
    ### YOUR CODE HERE

    ### END YOUR CODE
    pass
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE

    ### END YOUR CODE
    pass



import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F






import numpy as np
import pickle
N_img = 7 # number of train/test images
K = 50 # number of clusters

train_list = []

for i in range(N_img):
    train_list.append('train%d.jpg' % (i+1))
train_list
imgs=[cv2.imread(train_file,cv2.IMREAD_GRAYSCALE) for train_file in train_list]
img=io.imread(r"C:\Users\MV\Downloads\425_Assign\assign3\test1.jpg",as_gray=True)
img.shape

F.shape
F = makeLMfilters()
N_filters = F.shape[2]
N_filters
F.shape
F[:,:,0].shape
F[:,:,0]



import cv2
import numpy as np
### https://cs.brown.edu/courses/cs143/2011/results/proj2/mm28/
### https://www.geeksforgeeks.org/python-opencv-filter2d-function/
### https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
### https://www.cs.cmu.edu/~nileshk/projects/TextureClassification.pdf
### https://www.cs.auckland.ac.nz/~georgy/research/texture/thesis-html/node7.html#:~:text=Texture%20Classification,-Texture%20classification%20assigns&text=Two%20main%20classification%20methods%20are,characterisation%20for%20each%20texture%20class.
### https://www.youtube.com/watch?v=X-Y91ddBqaQ
### https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3
# Reading the image
image = cv2.imread(r'C:\Users\MV\Downloads\425_Assign\assign3\train3.jpg',cv2.IMREAD_GRAYSCALE)
image = cv2.imread(r'C:\Users\MV\Downloads\425_Assign\assign3\train4.jpg')

image = cv2.imread(r"C:\Users\MV\Downloads\missing.PNG")
imgs=[cv2.imread(train_file,cv2.IMREAD_GRAYSCALE) for train_file in train_list]
imgs[0]
# for i,fil in enumerate(F):
#     print(i)
# Creating the kernel(2d convolution matrix)
F = makeLMfilters()
numFil= F.shape[2]
F[:,:,1].shape
# Applying the filter2D() function
vec=[]
for i in range(numFil):
    vec.append(cv2.filter2D(src=image, ddepth=-1,kernel=F[:,:,i]))
vec=np.array(vec)
vec = np.transpose(vec, (1,2,0))
vec.shape
Verti = np.concatenate(vec, axis=0) 
xs,ys = np.random.(vec)
np.random(vec.shape[0])
vec.shape[0],vec.shape[1]
xs,ys=np.random.randint(0,vec.shape[0],[100]),np.random.randint(0,vec.shape[1],[100])
test=vec[xs,ys]
test.shape
test1=np.concatenate([test,test])
centroids, label, inertia = sklearn.cluster.k_means(X=test1,n_clusters=50)
len()
centroids.shape




################
image.shape
vec.shape
len(centroids)
image[0][0]
vec1=[]
for i in range(numFil):
    vec1.append(cv2.filter2D(src=image, ddepth=-1,kernel=F[:,:,i]))
vec1=np.array(vec1)
vec1.shape
vec1 = np.transpose(vec1, (1,2,0))
vec1.reshape(150*150,48).shape
vec2=vec1.reshape(150*150,48)
centroids,vec1[:,0,0]
res=cdist(vec2,centroids)
res.shape
res1=res.reshape((150,150,50))
res1.shape
len(res)
len(res[0])
[50,48] - [48,:,:]

vec1[:,0,0].shape
vec1.shape
centroids.shape
nearest_class_idx = np.argmin(res, axis=1)
nearest_class_idx.reshape(150,150).shape
nearest_class_idx.reshape(150,150)
nearest_class_idx.shape
nearest_class_idx[0]
hist = np.bincount(nearest_class_idx, minlength=50)

counts.shape
counts[0]



nearest_class_idx[1]
res
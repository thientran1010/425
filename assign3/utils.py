from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist
import cv2

def computeHistogram(img_file, F, textons):
    
    ### YOUR CODE HERE
    #Convert image to gray scale
    image = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    # Vector used to store numFil features
    vec1=[]
    # Number of features
    numFil= F.shape[2]
    for i in range(numFil):
        # Run matrix filter algorithm through each filter against the image
        vec1.append(cv2.filter2D(src=image, ddepth=-1,kernel=F[:,:,i]))
    #Transpose vector so that the filter vector is the last dimension (x,y,filter vector)
    vec1=np.array(vec1)
    # Transpose vector so that the filter vector is the last dimension (x,y,filter vector)
    vec1 = np.transpose(vec1, (1,2,0))
    # Get the dimension of the image
    D1,D2 = vec1.shape[0],vec1.shape[1]
    # Flatten the 2D image into 1D coordinates with vector of 48
    vec2=vec1.reshape(D1*D2,numFil)
    # Calculate the distance of each pixel against the centroids
    res=cdist(vec2,textons)
    # Assign each pixel to each centroid
    nearest_class_idx = np.argmin(res, axis=1)
    # Assign each pixel to each centroid bin
    hist = np.bincount(nearest_class_idx, minlength=50)


    ### END YOUR CODE
    return hist
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE
    
    # Convert link into images
    imgs=[cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in file_list]

    # Number of filters
    numFil= F.shape[2]

    # training array
    train_arr=[]

    for image in imgs:
        vec=[]
        for i in range(numFil):
            # Run matrix filter algorithm through each filter against the image
            vec.append(cv2.filter2D(src=image, ddepth=-1,kernel=F[:,:,i]))
        vec=np.array(vec)
        # Transpose vector so that the filter vector is the last dimension (x,y,filter vector)
        vec = np.transpose(vec, (1,2,0))
        # Select randomly 100 pixels
        xs,ys=np.random.randint(0,vec.shape[0],[100]),np.random.randint(0,vec.shape[1],[100])
        # 100 pixels, each with len(F) filter
        # Concatenated into a list
        train_arr.append(vec[xs,ys])
    # Convert a list into numpy list
    train_arr=np.concatenate(train_arr)
    # Train k_means algorithm
    centroids, label, inertia = sklearn.cluster.k_means(X=train_arr,n_clusters=K)
    # K centroids with len(F) length vector
    ### END YOUR CODE
    return centroids




















    




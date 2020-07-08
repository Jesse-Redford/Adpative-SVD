# SVD for Image Prcoessing and Compression
This repository contains guides, examples, and adpative methods for applying SVD compression to image and video files. Files include code for applying SVD-block compression scheme  to a varity of image types. Working towards building modules which allow a user to process a large quanity of images using SVD. 
The master function would allow you to specify a file path to your series of images, compression and PSNR threshold, followed by an output path for compressed files.Ideally this project would be used for deep learning and image recongnition training sets where one could train and test there models on both orginal and compressed data sets.


# Framework for Feature Oriented Compression
This project aims to highlight the research I have been conducting at UNCC in the mathmatrics deparment with Dr.Helen Xingje. In the code examples I have integrated OpenCV and matrix transformation techniques to enhance image and video compression by incorperating regions-of-intrests ROI's into the compression process.  If you would like to read more about my research feel free to read some of my manuscripts. 


![Fc-SVD algorithm](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/fc-SVD%20algorithm.PNG?raw=true)


# Background Removal

<details>
  <summary>Click here to see example code</summary>
 
```
# -*- coding: utf-8 -*-
# Background removal SVD

import sys
sys.path.append(r'C:\Users\Jesse\Desktop\OpenCV\openh264-1.6.0-win64msvc.dll')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import math

def k_svd(matrix,k=1):
    matrix = np.matrix(matrix)
    U,s,V = svd(matrix,full_matrices = False,compute_uv=True)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    #reconst_matrix = np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])
    reconst_matrix = reconst_matrix.clip(0)
    reconst_matrix = reconst_matrix.clip(max=255)
    return(reconst_matrix)

def block_svd(matrix,m = 2,n=2):
    matrix = np.matrix(matrix)
    M = matrix.shape[0]
    N = matrix.shape[1]
    Ak = np.zeros((M,N))
    for x in range(0, M, m):
        for y in range(0, N, n):
            block = matrix[x:x+m, y:y+n]
            Ak[x:x+m, y:y+n] = k_svd(block,k =1 ) # Fill int(min(M,N) * (((M*N)*np.sqrt(M**2 + N**2)) / ((M+N+1)*(M**2+N**2))))
    Ak = Ak.clip(0)
    Ak = Ak.clip(max=255)
    return Ak
    
    
####################  Define video files ######################################

codec = cv2.VideoWriter_fourcc('H','2','6','5')  # Codec to use
fps    =  30 
image = cv2.VideoWriter('Image.mp4v',codec, fps, (640,480),0)
figure = cv2.VideoWriter('Figure.mp4v',codec, fps, (640,480),0)
ground = cv2.VideoWriter('Ground.mp4v',codec, fps, (640,480),0)

################################################################################

scale_factor = 5
M = int(480/scale_factor)
N = int(640/scale_factor)


V = []

video_capture = cv2.VideoCapture(0)

plt.rcParams["axes.grid"] = True
while True:
    
    # read in frame from camera
    ret, frame = video_capture.read()
    
    # convert color frame to grayscale
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # scale aspect ratio of grayscale frame by factor of 10
    gs_frame = cv2.resize(gs_frame, (N, M))
    
    # Convert image to 1D vector in "Fortran" (column-major) order, ravel is faster since it does not create a copy in memory
    vec_gs_frame =   gs_frame.ravel('F')  # gs_frame.flatten('F') #
    
    # Recover orginal frame from vector representation 
    vec_to_gs_frame = vec_gs_frame.reshape(N, -1).T
    
    # Check that the orginal image was fully recovered, and show on screen
    assert (vec_to_gs_frame == gs_frame).all(), 'orginal image was not recovered'
    cv2.namedWindow('Recovered Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Recovered Frame',480,640)
    cv2.imshow('Recovered Frame', vec_to_gs_frame)
    
    # store vector frame in V
    V.append(vec_gs_frame)
    
    # press q to break while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    
# Convert V shape such that each video sample (frame vector) is a column of V
V = np.asarray(V).T
print('Shape of V:',V.shape) 
print('number of frames stored in V:',V.shape[1])

# Take Global low rank estimate of V
A_Gk = k_svd(V,k=3) 

# Take block based low rank estimate of V, where block dimenison n, is the resolution of estimate in time domain
frame_resolution = int(V.shape[1]/2)
A_Bk = block_svd(V,m =int(V.shape[0]/1),n=frame_resolution) 


# Plot V, notice the wavy lines represent movment (forground), and non changing horzontial lines are background (non-changing)
plt.imshow(V, cmap='hot')

plt.colorbar()
plt.show()

##V_fig = plt.figure(figsize=(10,10))
##ax1 = V_fig.add_subplot(1,2,1)
##ax2 = V_fig.add_subplot(1,2,2)
##ax1.imshow(V, cmap='hot')
##ax2.imshow(Ak, cmap='hot')
##V_fig.show()


# Loop through V, and seperate figure from ground
fig = plt.figure(figsize=(10,10))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(2,3,1)
ax1.set_title('Ground Truth')

ax2 = fig.add_subplot(2,3,2)
ax2.set_title('Global Low Rank Ground')

ax3 = fig.add_subplot(2,3,3)
ax3.set_title('Global Figure Approximation')

ax4 = fig.add_subplot(2,3,4)
ax4.set_title('Ground Truth')

ax5 = fig.add_subplot(2,3,5)
ax5.set_title('Block Low Rank Ground')

ax6 = fig.add_subplot(2,3,6)
ax6.set_title('Block Figure Approximation')

figs = []
for i in range(V.shape[1]):
    
    # show every other fram
    #if (i % 2) == 0:
        
    """ Global SVD """
    # orginal video
    I = V[:,i].reshape(N, -1).T
    ax1.imshow(I, cmap='gray',vmin=0, vmax=255)

    # low-rank background
    G = A_Gk[:,i].reshape(N, -1).T
    ax2.imshow(G, cmap='gray',vmin=0, vmax=255)
    
    # Figure "ROI"
    #F = (V[:,i].reshape(N, -1).T - A_Gk[:,i].reshape(N, -1).T)
    F = (A_Gk[:,i].reshape(N, -1).T - V[:,i].reshape(N, -1).T)
    ax3.imshow(F, cmap='gray',vmin=0, vmax=255)
    
    """ Block SVD """
    # orginal video
    I = V[:,i].reshape(N, -1).T
    ax4.imshow(I, cmap='gray',vmin=0, vmax=255)

    # low-rank background
    G_B = A_Bk[:,i].reshape(N, -1).T 
        
    ax5.imshow(G_B, cmap='gray',vmin=0, vmax=255)
    
    # Figure "ROI"
    #F_B = (V[:,i].reshape(N, -1).T - A_Bk[:,i].reshape(N, -1).T)
    F_B = (A_Bk[:,i].reshape(N, -1).T - V[:,i].reshape(N, -1).T)
    ax6.imshow(F_B, cmap='gray',vmin=0, vmax=255)
    
    plt.pause(0.00001)
    figs.append(fig)
    
fig.show()

```

</details>

# SVD_GUI.py
GUI application devloped with python, OpenCV, and PyQt5 which allows user to modify compression parameters while viewing the compressed video feed in real time.

![SVD GUI Example](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/SVD_GUI.PNG?raw=true)




# Dicom study 
File includes code for processing and displaying 2D and 3D reconstuctions of the pixel data within a DICOM image series.
The idea of this study is to determine wether SVD can be used to efficently compress and potentially enhance image quailty.


 
![DICOM Study, SVD compression effects on 3D reconstructions of CT and MRI scans](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/3D_Bone_Reconstruction_GIF.gif?raw=true)

<details>
  <summary>Click here to see example code</summary>

</details>




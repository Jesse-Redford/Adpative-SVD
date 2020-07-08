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




import os
import imageio

def make_giff(giff_name='My Giff',filetype_of_imgs='png',folder_name=None):
    
    # Assume that images are in current working directory
    if folder_name == None: 
      load_path = os.getcwd()
      
    # Otherwise search for path of specified folder containg the images 
    else:
        for root, dirs, files in os.walk(os.getcwd(), topdown=False):
            for name in dirs:  
                path = os.path.normpath(os.path.join(root, name)).split(os.sep) # Normalize path as proper str split path by '\' and store as array
                for sub_folders in path:
                    if sub_folders == folder_name:
                        load_path = os.path.join(root, name)
                      
                        print('The folder was found at:',os.path.join(root, name))
    
      
    images = []
    for file_name in os.listdir(load_path):
        if file_name.endswith('.'+filetype_of_imgs):
            file_path = os.path.join(load_path, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(giff_name+'.gif', images,duration = 1)
    return()


make_giff(giff_name='Name_of_giff',filetype_of_imgs = 'png',folder_name ='folder to load images from')

#make_giff(giff_name='My_giff',img_array=figs)




# close figures and save files     
video_capture.release()
cv2.destroyAllWindows()  



#plt.imshow(V, cmap='gray',vmin=0, vmax=255)
#plt.imshow(Ak, cmap='gray',vmin=0, vmax=255)
#plt.imshow(F, cmap='gray',vmin=0, vmax=255)


"""
# Loop through V, and seperate figure from ground
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

for i in range(len(V)):
    I = V[:,i].reshape(640, -1).T
    G = V[:,i].reshape(640, -1).T
    ax1.imshow(I, cmap='gray',vmin=0, vmax=255)
    ax2.imshow(G, cmap='gray',vmin=0, vmax=255)
    
    F = V[:,i]
    FP = V[:,i+1]
    FPP = V[:,i+2]
    FPP = V[:,i+2]
    
    
    if i > 1:
        FP = V[:,i-1]
        FPP = V[:,i+1]
    
    REFF = (F+FP+FPP)/3
    #REFF = (V[:,i]+V[:,i+1]+V[:,i+2]+V[:,i+3])/4
    
    for j in range(len(F)):
        if math.isclose(F[j],REFF[j], abs_tol=75) == False:
            F[j] = 0
  
         
    F = F.reshape(640, -1).T

    
    
    ax3.imshow(F, cmap='gray',vmin=0, vmax=255)
    
    plt.pause(0.1)
    

fig.show()
"""












#fig.show()
#ax1.plot(V)
#plt.pause(0.000001)
#plt.show()
#fig.tight_layout(pad=10.0, w_pad=10.0, h_pad=10.0)
#fig.show()   
#from mpl_toolkits.axes_grid.inset_locator import inset_axes
#fig = plt.figure(figsize=(10, 10))
#ax1 = plt.subplot2grid((1,2), (0,0)) 
#ax2 = plt.subplot2grid((1,2), (0,1)) 
    #cv2.imshow('image', B)
    
 
      

   # A = B.ravel() #gs_frame.flatten()
   # A = B.flatten('F')
    
    
    #A = A.reshape((48, 64),order='F')
    
    
    #print('color',frame.shape,frame.dtype)
    #print('gray',gs_frame.shape,gs_frame.dtype)
    #print('length A:',A.shape)
    

   #  final = cv2.vconcat([gs_frame, gs_frame])
     #    
   # image.write(gs_frame)
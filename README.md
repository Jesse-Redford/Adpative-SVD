# SVD
This repository contains guides, examples, and adpative methods for applying SVD compression to image and video files. If you would like to read more about my research feel free to read some of my manuscripts. 


<details>
  <summary>SVD library</summary>


</details>


# SVD Framework for Feature Oriented Compression
This project aims to integrate OpenCV and matrix transformation techniques to enhance image and video compression by incorperating regions-of-intrests ROI's into the compression process. 


<details>
  <summary>Click here to view concept for the algorithm</summary>
![Fc-SVD algorithm](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/fc-SVD%20algorithm.PNG?raw=true)
</details>

<details>
  <summary>Click here to see example code</summary>


</details>

# Dicom study
File includes code for processing and displaying 2D and 3D reconstuctions of the pixel data within a DICOM image series.
The idea of this study is to determine wether SVD can be used to efficently compress and potentially enhance image quailty.

<details>
  <summary>Click here to see example code</summary>
  
  #========================= DICOM Study ============================#

# FileName: svd_dicom_study.py
# Author: Jesse Redford
# Date: 12/22/2019

# Objective: Determine wether a Global & Localized rank approximation scheme(s) 
#            using SVD and CUR decomposition methods can reduce image noise and 
#            enhance 2D-3D image reconstruction of DICOM CT and MRI image series

#==================== Virtual Environment ==============================#

# Canopy Environment = admin = python 2.7.13
# Installation refrence for pydicom libary https://pydicom.github.io/pydicom/dev/old/getting_started.html
# Run script from virutal env
# ---> Open Canopy cmd
# ------> conda activate pydicomenv
# ---------> cd Desktop\Avante
# ------------> python dicom-python.py

#=====================Dependices-References=============================#
import os
import sys
sys.path.append(r'c:\users\jesse\appdata\local\programs\python\python36\lib\site-packages')

import pydicom 
from pydicom import dcmread
from pydicom import pixel_data_handlers

import numpy as np
from numpy.linalg import svd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as FF
from plotly.graph_objs import *

# ======================== Function Reference ========================#
# - convert_series_image(series_path,series_image):
# - load_scan(path):
# - get_pixels_hu(scans):
# - hist_plot(imgs_to_process):
# - sample_stack(stack, rows = ? , cols = ?, start_with = ?, show_every = ?):
# - resample(image, scan, new_spacing=[1,1,1]):
# - make_mesh(image, threshold, step_size):
# - plotly_3d(verts, faces, edges = True, slice_color_map = True): --> open interactive webpage with the 3D plot
# - plt_3d(verts, faces):

# SETUP & RUN A DICOM STUDY 

#===================== Load DICOM Data (Specify paths) ==================#
CT_Knee_Series1 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000001'
CT_Knee_Series2 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000002'
CT_Knee_Series3 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000003'
CT_Knee_Series4 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000004'
CT_Knee_Series5 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000005'
CT_Knee_Series6 = r'C:/Users/Jesse/Desktop/Avante/working_folder/CT/series-000006'
MRI_abdomen = r'C:\Users\Jesse\Desktop\Avante\Dicom studies-20191218T180250Z-001\Dicom studies\MRI\1b9baeb16d2aeba13bed71045df1bc65\series-000001'
series_image = 'image-000001.dcm' # specific image in series

#====================== Setup Directory  ==============================#
# Choose DICOM series to examine
data_path = MRI_abdomen 

# Select output path to save processed images
output_path = working_path = r'C:\Users\Jesse\Desktop\Avante\working_folder'

# Set a Reference ID
id=0

#====================== Process DICOM Image series  =====================#

# Process DICOM Series Slices
DICOM_Series = load_scan(data_path)

# Process and Extract Pixel Data, select number of components "k" for reconstruction
rank = k = 250
DICOM_Series_Images = get_pixels_hu(DICOM_Series,k)

# Save Processed Images to working_path
imgs_to_process = save_load_images(DICOM_Series_Images,id,output_path)

# Analysis Hounsfield Units - Need verify conversion and add better calibration criteria
#hist_plot(imgs_to_process)


#=============== 2D Reconstruction Visualization and Analysis  ================#

sample_stack(imgs_to_process) 
#convert_series_image(data_path,series_image) # display specific series image slice

#================= 3D Reconstruction Visualization and Analysis===============#

threshold = 225       # Filter --> MRI(bone~225,organs~100,skin~0) 
stepsize = 5         #  mesh --> fine = 1 < stepsize < coarse = 5,6,...
num_slices = imgs_to_process.shape[0]
edges = True          # set to true to see triangle 
slice_color_map = True # Set false for uniform color map

# Create 3D reconstruction
verts, faces = make_mesh(imgs_to_process, threshold,stepsize) 
plotly_3d(verts, faces, edges, slice_color_map,rank, threshold, stepsize, num_slices)


#======================= Function Definitions  ==========================#




def convert_series_image(series_path,series_image):
    ds = dcmread(os.path.join(series_path, series_image))
    print(ds.file_meta)
    print(ds.InstanceNumber)
    print(ds.ImageType)
    pix = ds.convert_pixel_data() 
    v = ds.pixel_array + -1024
    v = np.clip(v, 0, 50)
    plt.imshow(v, cmap='gray')
    plt.colorbar()
    plt.title('2000')
    plt.savefig('without_pillow.jpg')
    plt.show()


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices




def low_rank(matrix,k):
    U,s,V = svd(matrix,full_matrices = False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return(reconst_matrix)



def get_pixels_hu(scans,k):
    image = np.stack([s.pixel_array for s in scans])   # create stack pixel data from series scans
    image = np.stack([low_rank(i,k) for i in image]) # Apply rank-k approximation to each scan 
    image = image.astype(np.int16)                      # Convert to int16 - should be possible as values should always be low enough (<32k)
    image = image[::-1]                                 # reverse array (start for bottom of patient - orginal series order is from chest to pelvis
    image[image == -2000] = 0                           # Apply filter - Set outside-of-scan pixels to 1, 
                                                        # The intercept is usually -1024, so air is approximately 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept if 'RescaleIntercept' in scans[0] else -1024 # if rescaleintercept note aviable in DICOM file, set to -1024 to prevent error
    slope = scans[0].RescaleSlope if 'RescaleSlope' in scans[0] else 1                 # Same for rescaleslope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)



def hist_plot(imgs_to_process):
    plt.hist(imgs_to_process.flatten(), bins=10, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()




def sample_stack(stack, rows=2, cols=2, start_with=2, show_every=75): # 2,2,2,75 # use to plot samples of series slices 
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


    
def resample(image, scan, new_spacing=[1,1,1]): # need to fix, extracts pixel spacing between each slice from each DICOM file, will produce more accurate 3D mesh
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing)) # Determine current pixel spacing
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image #, new_spacing
    


def make_mesh(image, threshold, step_size): # great mesh from the image series, returns vertices and faces used for 3D plot
    print("Transposing surface")
    p = image.transpose(2,1,0)    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=False) # 
    return verts, faces



    

def plotly_3d(verts, faces, edges = True, slice_color_map = True, k = 0, threshold = 0, stepsize = 0, num_slices = 0): # Generate 3D reconstuction plot from mesh
    rank = str(k)
    threshold = str(threshold)
    stepsize = str(stepsize)
    num_slices = str(num_slices)
    if slice_color_map == True:
        colormap = "Portland"
    else:
        colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
        
    x,y,z = zip(*verts) 
    print("Drawing")
    
    fig = FF.create_trisurf(x=x,y=y,z=z, 
                        plot_edges=edges,
                        show_colorbar = True,
                        colormap = colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title= "Components used = " +rank+ "\n" + "Threshold = "+threshold + "\n" + "Stepsize = " + stepsize + "\n" + "Number of slices = " + num_slices)
    fig.show()



def save_load_images(images,id,output_path):
    np.save(output_path + "fullimages_%d.npy" % (id), images)
    file_used=output_path+"fullimages_%d.npy" % id
    imgs_to_process = np.load(file_used).astype(np.float64) 
    imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
    return(imgs_to_process)
    


</details>


# Image processing
File includes code for applying SVD-block compression and lowrank approximation to a varity of image types. 
Working towards building modules which allow a user to process a large quanity of images using SVD. 
The master function would allow you to specify a file path to your series of images, compression and PSNR threshold, followed by an output path for compressed files. Ideally this would be used for deep learning and image recongnition training sets where one could train and test there models on both orginal and compressed data sets.


# SVD

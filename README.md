# SVD
This repository contains guides, examples, and adpative methods for applying SVD compression to image and video files. If you would like to read more about my research feel free to read some of my manuscripts. Ideally this project would be used for deep learning and image recongnition training sets where one could train and test there models on both orginal and compressed data sets.

# SVD_GUI
GUI application devloped with python, OpenCV, and PyQt5 which allows user to modify compression parameters while viewing the compressed video feed in real time.

![SVD GUI Example](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/SVD_GUI.PNG?raw=true)


# SVD Framework for Feature Oriented Compression
This project aims to integrate OpenCV and matrix transformation techniques to enhance image and video compression by incorperating regions-of-intrests ROI's into the compression process. 


![Fc-SVD algorithm](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/fc-SVD%20algorithm.PNG?raw=true)


# Dicom study 
File includes code for processing and displaying 2D and 3D reconstuctions of the pixel data within a DICOM image series.
The idea of this study is to determine wether SVD can be used to efficently compress and potentially enhance image quailty.


 
![DICOM Study, SVD compression effects on 3D reconstructions of CT and MRI scans](https://github.com/Jesse-Redford/Adpative-SVD/blob/master/3D_Bone_Reconstruction_GIF.gif?raw=true)

<details>
  <summary>Click here to see example code</summary>

</details>


# Image processing
File includes code for applying SVD-block compression and lowrank approximation to a varity of image types. 
Working towards building modules which allow a user to process a large quanity of images using SVD. 
The master function would allow you to specify a file path to your series of images, compression and PSNR threshold, followed by an output path for compressed files. Ideally this would be used for deep learning and image recongnition training sets where one could train and test there models on both orginal and compressed data sets.


# SVD

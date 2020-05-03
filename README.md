# SVD
This repository contains guides, examples, and adpative methods for applying SVD compression to image and video files. 


# Dicom study
File includes code for processing and displaying 2D and 3D reconstuctions of the pixel data within a DICOM image series.
The idea of this study is to determine wether SVD can be used to efficently compress and potentially enhance image quailty.


# Image processing
File includes code for applying SVD-block compression and lowrank approximation to a varity of image types. 
Working towards building modules which allow a user to process a large quanity of images using SVD. 
The master function would allow you to specify a file path to your series of images, compression and PSNR threshold, followed by an output path for compressed files. Ideally this would be used for deep learning and image recongnition training sets where one could train and test there models on both orginal and compressed data sets.


# SVD

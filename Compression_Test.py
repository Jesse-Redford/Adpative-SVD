# -*- coding: utf-8 -*-

# File: SVD_Video_Compression_Research
# Author: Jesse Redford
# Date: 4/9/2020

###############################################################################
#------------------------------ System Setup ---------------------------------#
###############################################################################

# pip3 install face_recognition - verison 1.2.3
# pip install opencv-python - version 4.2.0

import sys
sys.path.append(r'C:\Users\Jesse\Desktop\OpenCV\openh264-1.6.0-win64msvc.dll')
import os
import cv2
import face_recognition
import numpy as np
import time 
import random
import SVD_comp_lib
from SVD_comp_lib import k_svd
from SVD_comp_lib import block_svd
from SVD_comp_lib import get_divisors
from SVD_comp_lib import compression_ratio
from SVD_comp_lib import error_metrics 
from SVD_comp_lib import figure_block_svd

print('Python Version', sys.version)
print('OpenCV Version',cv2.__version__)
print('Face recogntion Version', face_recognition.__version__)

###############################################################################
#------------------------  User Parameters  ----------------------------------#
#-----------------------------------------------------------------------------#
# Video Codec Types
# - uncompressed YUV      -----> 'I','4','2','0'   all vid sizes will be same
# - compressed H.264      -----> 'H','2','6','4'  .mp4v file type
# - compressed H.265      -----> 'H','2','6','5'  .mp4v file type
# - compressed MPGEG-4    -----> 'X','V','I','D'  .avi mp4  (good compression)
# - compressed mjpg       -----> 'M','J','P','G'  .avi mp4
# - compressed MPEG-1     -----> ‘P’,’I’,’M’,’1’  .avi

#------------------------------------------------------------------------------#
codec = cv2.VideoWriter_fourcc('H','2','6','5')  # Codec to use

frames =  5     # number of frames you want to record
fps    =  3      # frames-per-second for avi files
M      =  480    # row size of incoming video frame
N      =  640    # row size of incoming video frame
k      =  50     # rank for g_svd
mb     =  8     # row block size for b_svd
nb     =  8     # colmn block size for b_svd
m      =  8     # row block size for ab_svd
n      =  8     # column block size for ab_svd

ms,ns     =  get_divisors(np.zeros(shape=(M,N))) # ms,ns list of divisors
font      =  cv2.FONT_HERSHEY_SIMPLEX            # text font for results
color     =  (0, 0, 255)                         # font color (0,0,255) = black
org       =  (10,25)                             # location of image metrics
org_fig   =  (10,50)                             # location of figure metrics
fontScale =   .35                                 # size of text 
thickness =    1                                 # thickness of text, int type

window_height = 960   # size of display window
window_width  = 1280  # size of display window

################################################################################

video_capture = cv2.VideoCapture(0)

#result = cv2.VideoWriter('0_results.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (1280,960),0)
result = cv2.VideoWriter('0_results.mp4v',cv2.VideoWriter_fourcc('I','4','2','0'), fps, (1280,960),0)
gs     = cv2.VideoWriter('1_gs.mp4v',codec, fps, (640,480),0)
g_svd  = cv2.VideoWriter('2_g_svd.mp4v',codec, fps, (640,480),0)
b_svd  = cv2.VideoWriter('3_b_svd.mp4v',codec, fps, (640,480),0)
ab_svd = cv2.VideoWriter('4_ab_svd.mp4v',codec, fps, (640,480),0)

gs_face_locations = []
g_svd_face_locations = []
b_face_locations = []
ab_svd_face_locations = []
rate = time.time()
for i in range(frames): 
    print(rate-time.time())
    ret, frame = video_capture.read()
    #if not ret: break
    
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #print(frame.shape,frame.dtype)
    
    start = time.time() 
    g_svd_frame  = np.uint8(k_svd(gs_frame,k))
    end = time.time()
    g_svd_time = round(end-start,2) 
    
    start = time.time()
    b_svd_frame  = np.uint8(block_svd(gs_frame,mb,nb))
    end = time.time()
    b_svd_time = round(end-start,2)
    
    for x1, y1, x2, y2 in face_recognition.face_locations(gs_frame):
      
        start = time.time() 
        ab_svd_frame = np.uint8(figure_block_svd(gs_frame,m,n,x1,x2,y1,y2)) # adpate for multiple faces
        end = time.time()
        ab_svd_time = round(end-start,2)
        
    
    g_svd_CF  =  round((480 * 640) / (k*(1+480+640)),2)
    b_svd_CF  =  round(compression_ratio(M,N,mb,nb),2)
    ab_svd_CF =  round(compression_ratio(M,N,m,n),2) # add calculation for varying k

    g_svd_RMSE,g_svd_PSNR   = error_metrics(gs_frame,g_svd_frame)
    b_svd_RMSE,b_svd_PSNR   = error_metrics(gs_frame,b_svd_frame)
    ab_svd_RMSE,ab_svd_PSNR = error_metrics(gs_frame,ab_svd_frame)
    fig_ab_svd_RMSE,fig_ab_svd_PSNR = error_metrics(gs_frame[x1:x2,y2:y1],ab_svd_frame[x1:x2,y2:y1])

    gs_face_locations     = face_recognition.face_locations(gs_frame)
    g_svd_face_locations  = face_recognition.face_locations(g_svd_frame)
    b_svd_face_locations  = face_recognition.face_locations(b_svd_frame)
    ab_svd_face_locations = face_recognition.face_locations(ab_svd_frame)
    
    g_svd_text = 'Matrix Rank:' + str(k) + ' | '              \
                + 'CF:' + str(g_svd_CF) + ' | '                \
                + 'TC:' + str(g_svd_time) + ' | '               \
                + 'RMSE:' + str(round(g_svd_RMSE,2)) + ' | '    \
                + 'PSNR:' + str(round(g_svd_PSNR,2)) + ' | '     \
             
    b_svd_text = 'Block Rank:' + str(1) + ' | '                  \
                + 'CF:' + str(b_svd_CF) + ' | '                   \
                + 'TC:' + str(b_svd_time) + ' | '               \
                + 'RMSE:' + str(round(b_svd_RMSE,2)) + ' | '       \
                + 'PSNR:' + str(round(b_svd_PSNR,2)) + ' | '        \
                + 'Blocksize:' + str(mb) +'x'+str(nb) + ' |'
                
    ab_svd_text = 'Ground Rank:' + '1' + ' | '                      \
                + 'CF:' + str(ab_svd_CF) + ' | '                     \
                + 'TC:' + str(ab_svd_time) + ' | '               \
                + 'RMSE:' + str(round(ab_svd_RMSE,2)) + ' | '         \
                + 'PSNR:' + str(round(ab_svd_PSNR,2)) + ' | '          \
                + 'Blocksize:' + str(m) +'x'+str(n) + ' |'
    
    ab_svd_fig_text = 'Figure Rank:' + 'adpative varies' + ' | '        \
                    + 'RMSE:' + str(round(fig_ab_svd_RMSE,2)) + ' | '    \
                    + 'PSNR:' + str(round(fig_ab_svd_PSNR,2)) + ' | '     \
                  
    gs_frame     = cv2.putText(gs_frame,'Ground Truth', org, font, fontScale,color,thickness, cv2.LINE_AA, False)       
    g_svd_frame  = cv2.putText(g_svd_frame,str(g_svd_text), org, font, fontScale, color,thickness, cv2.LINE_AA, False)      
    b_svd_frame  = cv2.putText(b_svd_frame,str(b_svd_text), org, font, fontScale, color,thickness, cv2.LINE_AA, False)       
    ab_svd_frame = cv2.putText(ab_svd_frame,str(ab_svd_text), org, font, fontScale, color,thickness, cv2.LINE_AA, False)
    ab_svd_frame = cv2.putText(ab_svd_frame,str(ab_svd_fig_text), org_fig, font, fontScale, color, thickness, cv2.LINE_AA, False)
                          
    for x1, y1, x2, y2 in gs_face_locations:
        cv2.rectangle(gs_frame, (y1, x1), (y1+(y2-y1), x1+(x2-x1)), (0, 255, 0), 2)  
             
    for x1, y1, x2, y2 in g_svd_face_locations:
        cv2.rectangle(g_svd_frame, (y1, x1), (y1+(y2-y1), x1+(x2-x1)), (0, 255, 0), 2)
        
    for x1, y1, x2, y2 in b_svd_face_locations:
        cv2.rectangle(b_svd_frame, (y1, x1), (y1+(y2-y1), x1+(x2-x1)), (0, 255, 0), 2)
        
    for x1, y1, x2, y2 in ab_svd_face_locations:
        cv2.rectangle(ab_svd_frame, (y1, x1), (y1+(y2-y1), x1+(x2-x1)), (0, 255, 0), 2)
        
    stream1      = np.concatenate((gs_frame,g_svd_frame), axis=1)
    stream2      = np.concatenate((b_svd_frame,ab_svd_frame), axis=1)
    multi_stream = np.concatenate((stream1,stream2), axis=0) # print('streams',streams.shape) make sure videowriter has correct shape
    
    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results',window_width,window_height)
    cv2.imshow('Results',multi_stream)
    
    result.write(multi_stream)
    gs.write(gs_frame) 
    g_svd.write(g_svd_frame) 
    b_svd.write(b_svd_frame) 
    ab_svd.write(ab_svd_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
      
video_capture.release()
gs.release()
g_svd.release()
b_svd.release()
ab_svd.release()
result.release()


cv2.destroyAllWindows()   
        
        
        
    

        
        
        
        
        
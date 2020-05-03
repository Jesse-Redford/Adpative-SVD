import sys
#print(sys.path)
#sys.path.append(r'c:\users\timot\appdata\local\programs\python\python37\lib\site-packages')
sys.path.append(r'C:\Users\Jesse\Desktop\Compression  Program & Research\Programs\Backups\Python_SVD')


import PIL
from PIL import Image
from pylab import *
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Times New Roman"

import sewar
from sewar import*
import itertools
from itertools import product
import pandas as pd



def get_dim(matrix):
    M = np.shape(matrix)[0] # Row
    N = np.shape(matrix)[1] # Column
    return(M,N)

def get_divisors(matrix):
    
    M, N = get_dim(matrix)
    
    M_div = []
    N_div = []
    
    for i in range(1,N+1):
        if(N%i==0):
            N_div.append(i)
    for i in range(1,M+1):
        if(M%i==0):
            M_div.append(i)
            
    M_div.pop(0)
    N_div.pop(0)
        
    return(M_div, N_div)


def error_metrics(A,Ak):
    RMSE, MAP = sewar.full_ref.rmse_sw(A, Ak, ws=8)
    PSNR = sewar.full_ref.psnr(A, Ak, MAX=None)
    return(RMSE,PSNR)

def compression_ratio(M = 1,N = 1 ,m = 1,n = 1):
    iters = (M/m) * (N/n)
    block_memory =  (1+m+n)
    compressed_size = iters * block_memory
    uncompressed_size = M * N
    CF = uncompressed_size / compressed_size
    return(CF)
    
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def k_svd(matrix,k):
    #print(matrix)
    #matrix = im2double(matrix)
    U,s,V = svd(matrix,full_matrices = False,compute_uv=True)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    #reconst_matrix = np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])
    reconst_matrix = reconst_matrix.clip(0)
    reconst_matrix = reconst_matrix.clip(max=255)
    return(reconst_matrix)

def LRA(matrix):
    M, N = get_dim(matrix)
    rank = min(M,N)
    LRA = int(rank * (((M*N)*np.sqrt(M**2 + N**2)) / ((M+N+1)*(M**2+N**2))))
    LRA_matrix = k_svd(matrix,LRA)
    return(LRA_matrix)
    
def block_svd(matrix,m = 2,n=2):
    M, N = get_dim(matrix)
    Ak = np.zeros((M,N))
    for x in range(0, M, m):
        for y in range(0, N, n):
            block = matrix[x:x+m, y:y+n]
            Ak[x:x+m, y:y+n] = k_svd(block,k=1 ) # Fill 
    return Ak
    
def revised_block_svd(matrix,m = 2,n=2):
    M, N = get_dim(matrix)
    #Ak = np.zeros((M,N))
    for x in range(0, M, m):
        for y in range(0, N, n):
            block = matrix[x:x+m, y:y+n]
            matrix[x:x+m, y:y+n] = k_svd(block,k=1 ) # Fill
            
    return matrix # returned image is differnt but zero error between orginal???



def figure_block_svd(matrix,m,n,x1,x2,y1,y2):
    M, N = get_dim(matrix)
    Ak = np.zeros((M,N))
    for x in range(0, M, m):
        for y in range(0, N, n):
            block = matrix[x:x+m, y:y+n]
            if x1 < x+m < x2+m and y2 < y+n < y1+n: 
                k = int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2))))
            else:
                k = 1 #int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2)))/2)
            Ak[x:x+m, y:y+n] = k_svd(block,k) # Fill 
            
    return Ak
    
    

def Fc_SVD(matrix,M,N,m,n,face_locations):
    start = time.time()
    #M, N = get_dim(matrix)
    Ak = np.uint8(np.zeros((M,N)))
    X1=[]
    X2=[]
    Y1=[]
    Y2=[]
    for i in range(len(face_locations)):
        x1, y1, x2, y2 = face_locations[i]
        X1.append(x1)
        Y1.append(y1)
        X2.append(x2)
        Y2.append(y2)
    for x in range(0, M, m):
        for y in range(0, N, n):
            block = matrix[x:x+m, y:y+n]
            fig = False
            for i in range(len(face_locations)):
                if X1[i]-m < x+m/2 < X2[i]+m and Y2[i]-n < y+n/2 < Y1[i]+n:
                #if X1[i] < x+m < X2[i]+m and Y2[i] < y+n < Y1[i]+n:
                    fig = True
            if fig == True: 
                k = int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2))))
                if k == 0:
                    k =1
            else:
                k=1
            Ak[x:x+m, y:y+n] = k_svd(block,k)
         
    Ak = np.uint8(Ak)
    end = time.time()
    PT = round(end-start,2) # processing time
    TC = time_complexity(M,N,m,n)
    
    G_RMSE,G_PSNR = error_metrics(matrix,Ak)
    
    RMSE = []
    PSNR = []
    for x1, y1, x2, y2 in face_locations:
        F_RMSE,F_PSNR = error_metrics(matrix[x1:x2,y2:y1],Ak[x1:x2,y2:y1])
        RMSE.append(F_RMSE)
        PSNR.append(F_PSNR)
    
    F_RMSE = np.mean(RMSE)
    F_PSNR = np.mean(PSNR)
    
    CF =  round(compression_ratio(M,N,m,n),2)
        
    figure_metrics = 'Figure Rank:' + 'adpative varies' + ' | '    \
                    + 'RMSE:' + str(round(F_RMSE,2)) + ' | '    \
                    + 'PSNR:' + str(round(F_PSNR,2)) + ' | '     \
            
    ground_metrics = 'Ground Rank:' + '1' + ' | '                      \
                + 'CF:' + str(CF) + ' | '                     \
                + 'TC:' + str(PT) + ' | '               \
                + 'RMSE:' + str(round(G_RMSE,2)) + ' | '         \
                + 'PSNR:' + str(round(G_PSNR,2)) + ' | '          \
                + 'Blocksize:' + str(m) +'x'+str(n) + ' |'
                
                
    return Ak,figure_metrics,ground_metrics
    
    
    



# ---------- orignial  -----------------------#
#def figure_block_svd(matrix,m,n,x1,x2,y1,y2):
 #   M, N = get_dim(matrix)
  #  Ak = np.zeros((M,N))
   # for x in range(0, M, m):
    #    for y in range(0, N, n):
     #       block = matrix[x:x+m, y:y+n]
      #      if x1 < x+m < x2+m and y2 < y+n < y1+n: 
       #         k = int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2))))
        #    else:
         #       k = 1 #int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2)))/2)
          #  Ak[x:x+m, y:y+n] = k_svd(block,k) # Fill 
    #return Ak
#---------------------------------------#
def time_complexity(M,N,m,n):
    iters = (M/m) * (N/n)
    svd_tc = max(m,n)**2 * min(m,n)
    tc = iters * svd_tc
    return tc

def get_metics(A,Ak,M,N,m,n):
    bs = n
    cr =  compression_ratio(M,N,m,n)
    ss = space_savings(cr)
    tc = time_complexity(M,N,m,n)
    rmse, psnr = error_metrics(A,Ak)
    
    return(bs,cr,ss,tc,rmse,psnr)

def get_block_size(block_combinations, index):
    tup = block_combinations[index]
    m = tup[0]
    n = tup[1]
    return(m,n)

def plot_images(image_list):
    fig = plt.figure(figsize=(20,10))
    for i in range(len(image_list)):
        ax=fig.add_subplot(1,len(image_list),i+1)
        ax.imshow(image_list[i], cmap=cm.gray)
        plt.tight_layout()
    return()
    
def space_savings(CF):
    ss = (1-1/CF)*100
    return(ss)

def SS_vs_PSNR(df):
# Dual Axis SS + PSNR
    fig = plt.figure(figsize=(5,5))
    ax1 =  df.plot(x='BS', y='SS', style='g^', grid = True)
    ax1.legend()
    plt.xlabel("Block Size")
    plt.ylabel("Space Savings %")
    ax2 = df['PSNR'].plot(secondary_y=True, mark_right=False, style='r^', grid = True) #color='k', marker='o')
    ax2.legend(loc=1)
    ax2.set_ylabel('PSNR db')
    plt.show()
    plt.savefig('Test_Graphs.png')
    return()


def CF_vs_SS(df):
# Dual Axis CF + SS
    ax1 =  df.plot(x='BS', y='CR', style='g^', grid = True)
    ax1.legend()
    plt.xlabel("Block Size")
    plt.ylabel("Compression Ratio")
    ax2 = df['SS'].plot(secondary_y=True, mark_right=False, style='r^', grid = True) #color='k', marker='o')
    ax2.legend(loc=1)
    ax2.set_ylabel('Space Savings')
    plt.show()
    return()


def plot_metrics(df):
    fig = plt.figure(figsize=(5,5))
    #df.plot(subplots=True, layout=(5,5),sharex=True)
    
    #CF_vs_SS(df)
    
    df.plot(x ='BS', y='SS', style='r^', grid = True)
    plt.xlabel("Block Size")
    plt.ylabel("Space Savings")
    
    df.plot(x ='BS', y='TC', style='r^', grid = True)
    plt.xlabel("Block Size")
    plt.ylabel("Time Complexity")
    
    df.plot(x ='BS', y='RMSE', style='b^', grid = True)
    plt.xlabel("Block Size")
    plt.ylabel("RMSE")
    
    df.plot(x ='BS', y='PSNR', style='bs', grid = True)
    plt.xlabel("Block Size")
    plt.ylabel("PSNR db")
    
    plt.tight_layout()
    plt.savefig('Test_Graphs.png')
    plt.show()
    return()



def create_table(df):
    # Create Table Figure
    from pandas.plotting import table

    # set fig size
    fig, ax = plt.subplots(figsize=(7.5, 15)) 
    # no axes
    ax.xaxis.set_visible(False)  
    ax.yaxis.set_visible(False)  
    # no frame
    ax.set_frame_on(False)  
    # plot table
    tab = table(ax, df, loc='center')  
    # set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(10) 
    #plt.title('Loss by Disaster')
    # save the result
    plt.savefig('table.png')
    return()


# Filtered Data 
def sort_results(df):

    filtered_df = df[(df['CR'] >= 2) & (df['PSNR db'] >= 30)]
    filtered_df.sort_values(by=['PSNR db'], ascending=False)
    filtered_df.round({'CR': 2, 'TC': 0,'RMSE': 2, 'PSNR db': 2 })

    print(filtered_df)
    #filtered_df.plot(x ='BS', y='CR', style='bs', grid = True)
    #filtered_df.plot(x ='BS', y='PSNR db', style='bs', grid = True)
    return(filtered_df)



#plot_images(images)

def display_select_images(A, images, index1, index2, index3):
    fig = plt.figure(figsize=(5,5))

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(A, cmap=cm.gray)
    ax1.set_xlabel('(a)')

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(images[index1],cmap=cm.gray)
    ax2.set_xlabel('(b)')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(images[index2], cmap=cm.gray)
    ax3.set_xlabel('(c)')

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(images[index3], cmap=cm.gray)
    ax4.set_xlabel('(d)')

    plt.tight_layout()

    fig.savefig('Test_py_plot.png')
    return()




def run_test():

    # Input image and convert to array
    #A = array(Image.open(r'C:\Users\timot\Desktop\lenna_greyscale.png'))
    #A = array(Image.open(r'C:\Users\timot\Desktop\peppers.png'))
    #A = array(Image.open(r'C:\Users\timot\Desktop\boat.png'))
    A = array(Image.open(r'C:\Users\Jesse\Desktop\Compression  Program & Research\Programs\Backups\Python_SVD\turtle.jpg'))
    A = A[:,:,0]

    # Store results
    blk_m = []
    blk_n =[]
    BS = []
    CR = []
    SS = []
    TC = []
    RMSE = []
    PSNR = []
    images = []

    # determine image size and blocksize combinations
    M, N = get_dim(A)
    mi, ni = get_divisors(A)
    block_combinations = list(itertools.product(mi,ni))

    # process image using every blocksize combination with a rank 1 approximation, 
    # reconstruct images and record preformance metrics
    i = 1
    while i < len(block_combinations):
    
        m,n = get_block_size(block_combinations, i)

        Ak = np.uint8(block_svd(A,m,n))
    
        bs,cr,ss,tc,rmse,psnr = get_metics(A,Ak,M,N,m,n)
    
        images.append(Ak)
        blk_m.append(m)
        blk_n.append(n)
        BS.append([m,n])
        CR.append(round(cr,2))
        SS.append(round(ss,2))
        TC.append(tc)
        RMSE.append(round(rmse,2)) 
        PSNR.append(round(psnr,2))
    
        i = i+1

    # orginize data into a dataframe table
    data = {'BS':BS,'CR':CR,'SS':SS,'PSNR':PSNR,'RMSE':RMSE,'TC':TC}
    df = pd.DataFrame(data)

    # Create new data table sorted by PSNR
    df_PSNR = df.sort_values(by=['PSNR'], ascending=False)
    df_CR = df.sort_values(by=['CR'], ascending=False)

    df_filter = df[(df['CR'] >= 4) & (df['PSNR'] >= 27)]

    #plot_metrics(df_PSNR)
    create_table(df_CR)
    SS_vs_PSNR(df_PSNR)
    display_select_images(A, images, index1 = 30, index2 = 29, index3 = 73)

    # index 73,25, 
    #rmse, psnr = error_metrics(A,images[73])
    return()



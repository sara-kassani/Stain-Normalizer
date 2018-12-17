import numpy as np
from scipy import ndimage, misc
import sys
import skimage
from skimage import feature, data, io
from skimage.morphology import disk
from scipy.misc import imread, imsave
import util.dirhandler
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters


def normalize_staining (I, Io=240, beta=0.15, alpha=1, HERef=None, maxCRef=None):
    '''
    Normalize the staining appearance of images originating from hematoxylin-eosin (H&E) stained sections.

    Inputs:
        I       - Input image (cmap=RGB, dtype=float64)
        Io      - Tra    print('\tfiltering...')nsmitted light intensity (default: 240)
        beta    - OD threshold for transparent pixels (default: 0.15)
        alpha   - Tolerance for the pseudo-min and pseudo-max (default: 1)
        HERef   - Reference H&E OD matrix
        maxCRef - Reference maximum stain concentrations for H&E

    Output:
        I_norm  - Normalized image
        H_cmp   - Hematoxylin component image
        E_cmp   - Eosin component image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009

    Support:
        https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m

    '''

    if HERef is None:
        HERef = np.array([[0.5626,0.2159],[0.7201,0.8012],[0.4062,0.5581]])

    if maxCRef is None:
        maxCRef = np.array([1.9705,1.0308])

    (h,w,d) = I.shape

    I = I.reshape((h*w,d),order='F')

    # Convert RGB to optical density (OD)
    OD = -np.log((I+1)/Io)

    # Remove data with negligible optical density (OD < beta)
    ODhat = OD[(np.logical_not((OD < beta).any(axis=1))),:]

    # Calculate SVD on the OD tuples
    (W,V) = np.linalg.eig(np.cov(ODhat,rowvar=0))

    # Project data onto the plane obtained from the SVD directions corresponding to the two largest singular values
    vectors = np.array([V[:,1],V[:,0]])
    vectors = vectors.T
    That = np.dot(ODhat, vectors)

    # Calculate angle of each point wrt the first SVD direction
    phi = np.arctan2(That[:,1], That[:,0])

    # Find robust extremes (1st and 99th percentiles) of the angle
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    # Convert extreme values back to OD space --> Optimal Stain Vectors
    vMin = np.dot(vectors, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(vectors, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
    if vMin[0]>vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])
    HE = HE.T

    # Rows correspond to channels (RGB), columns to OD values
    Y = OD.reshape((h*w,d))
    Y = Y.T

    # Determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y)

    # Normalize stain concentrations
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0]/maxC[:,None]
    C = C*maxCRef[:,None]

    # Recreate the image using reference mixing matrix
    I_norm = Io*np.exp(- np.dot(HERef,C))
    I_norm = np.reshape(I_norm.T, (h,w,d), order='F')
    I_norm = np.clip(I_norm,0,255)
    I_norm = I_norm.astype(np.uint8)

    HERef_0 = HERef[:,0].reshape(((HERef[:,0]).shape[0],1))
    C_0 = C[0,:].reshape(1,(C[0,:]).shape[0])
    H_comp = Io*np.exp(- np.dot(HERef_0, C_0))
    H_comp = np.reshape(H_comp.T, (h,w,d), order='F')
    H_comp = np.clip(H_comp,0,255)
    H_comp = H_comp.astype(np.uint8)

    HERef_1 = HERef[:,1].reshape(((HERef[:,1]).shape[0],1))
    C_1 = C[1,:].reshape(1,(C[1,:]).shape[0])
    E_comp = Io*np.exp(- np.dot(HERef_1, C_1))
    E_comp = np.reshape(E_comp.T, (h,w,d), order='F')
    E_comp = np.clip(E_comp,0,255)
    E_comp = E_comp.astype(np.uint8)

    return I_norm, H_comp, E_comp




"""
Get name images to preprocessing
--------------------------------
"""
input_path = 'data/validation/Benign/'
input_extension = 'tif'

inputs_files = sorted(util.dirhandler.get_file_name_dir(input_path, input_extension))

print('*************************************************************')
print('********************* Preprocessing *************************')
print('*************************************************************')

"""
Iterate over the list of images
-------------------------------
"""
cont = 1

for file_name in inputs_files:
    print('Preprocessing: ' + file_name + ': ' + str(cont))
    cont += 1
    # Get the image
    # image = imread(input_path + file_name)
    rgb_img = np.array(Image.open(input_path + file_name),dtype=np.float64)
    img_normalized, H_img, E_img = normalize_staining(rgb_img)

    imsave('data/processed-data/validation/Benign/' + file_name, img_normalized)

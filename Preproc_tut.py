import numpy as np      # linear algebra
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sys
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#TODO create a user propt and file selector to enable the user to choose a file/files containing the DICOM
INPUT_FOLDER = '/Users/danielhardej/Documents/Med_Image_Preprocessing/Preprocessing_Tut/input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# Load files from imaging scan from given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

# Assign the appropriate Houndsfield Unit to the pixel values of the images
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to pixel value to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

# Test function using files fro first patient
first_patient = load_scan(INPUT_FOLDER + patients[1])
print('Loading fist patient scans...')
first_patient_pixels = get_pixels_hu(first_patient)
print('Loading fist patient image data...')
plt.figure(1, figsize=(10,5))
plt.subplot(121)
plt.title('Frequency distribution of Hounsfield Unit value')
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
# Show some slice in the middle in a second window
plt.subplot(122)
plt.title('Middle image slice')
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
# Display both figures
plt.show()

close = input('Continue? (Y/N) ')
if close == 'N':
    sys.exit()
else:
    pass

def resample(image, scan, new_spacing=[1,1,1]):
    """function to assign new pixel spacing of 1mmx1mmx1mm in order to achieve
    an isotropic resolution for all image sets, since some scans show varition
    in pixel spacing, which causes difficulties in automatic analysis using
    convolutional neural nets!"""
    # First determine original pixel spacing
    original_spacing = np.array([scan[0].SliceThickness]+scan[0].PixelSpacing, dtype=np.float32)
    # Then compute the new shape and pixel spacing
    resize_factor =      original_spacing / new_spacing
    new_real_shape =     image.shape * resize_factor
    new_shape =          np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing =        original_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling: ", first_patient_pixels.shape)
print("Shape after resampling:  ", pix_resampled.shape)

def plot_3d(image, threshold=-300):
    """This function creates a 3D plot of the patient scan. The 3D plot is positioned
    to be upright."""
    # Scan is positioned upright by transposing the image
    p =             image.transpose(2,1,0)
    verts, faces =  measure.marching_cubes_classic(p, threshold)
    fig =           plt.figure(figsize=(5,5))
    ax =            fig.add_subplot(111,projection='3d')
    mesh =          Poly3DCollection(verts[faces], alpha=0.70)
    face_color =    [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

#plot_3d(pix_resampled, 400)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

segmented_lungs = segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

# plot_3d(segmented_lungs, 0)
# plot_3d(segmented_lungs_fill, 0)
# plot_3d(segmented_lungs_fill - segmented_lungs, 0)

# Assign HU threshold
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
# The last two preprocessing stages are normalization and zero-centering
#
def normalize(image):
    """This function takes the range of suitable HU values from the image and
    normalizes the distribution of values according to predefined minimum and
    maximum bounds"""
    # Assign new values to image
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1
    image[image<0] = 0
    return image

pixel_mean = 0.25
def zero_center(image):
    image = image - pixel_mean
    return image

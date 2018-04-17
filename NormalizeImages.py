import dicom

class NormalizeImage():
    """"""
    def __init__(self, arg):


    def get_pixels_hu(slices):
        """Function to ssign the Houndsfield Unit to the pixel values of the images"""
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


    def resample(image, scan, new_spacing=[1,1,1]):
        """function to assign new pixel spacing of 1mmx1mmx1mm in order to achieve
        an isotropic resolution for all image sets
        some scans show varition in pixel spacing, which causes difficulties in
        automatic analysis using convolutional neural nets and other classifiers."""
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

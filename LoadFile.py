import dicom

class LoadFile():
    """
    loadFile()
        Opens a series of DICOM files.
    """
    def __init__(self, arg):
        self.path = path
        pass

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

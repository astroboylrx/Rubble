import traceback
import struct
from array import array
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt


def readbin(file_handler, dtype='d'):
    """ Retrieve a certain length of data from a binary file
    :param file_handler: an opened file object
    :param dtype: data type, format string
    :return: re-interpreted data
    """

    if not isinstance(dtype, str):
        raise TypeError("reading format need to be a str: ", dtype)

    ori_pos = file_handler.tell()
    try:
        data = struct.unpack(dtype, file_handler.read(struct.calcsize(dtype)))
        if len(data) == 1:
            return data[0]
        return data
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return None

def loadbin(file_handler, dtype='d', num=1):
    """ Load a sequence of data from a binary file, return an ndarray """

    __valid_array_typecode = ['b', 'B', 'u', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'f', 'd']
    if dtype not in __valid_array_typecode:
        raise ValueError("bad typecode: "+dtype+" (must be one of ["+",".join(__valid_array_typecode)+"])")
    if not isinstance(num, Number):
        raise TypeError("size should be a scalar, but got: ", num)
    if num < 0:
        raise ValueError("size should be larger than 0, but got: "+str(num))

    data = array(dtype)
    ori_pos = file_handler.tell()
    try:
        data.fromfile(file_handler, num)
        data = np.asarray(data)
        return data
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return None

class RubbleData:
    """ Read in simulation data for further analyses """

    def __init__(self, filename, data_format='bin'):
        """
        Read in simulation data for further analyses
        
        Parameters
        ----------
        filename : str
            name of the data file
        data_format : str
            'bin': the data is in binary format (by default)
            'ascii': the data is in ascii data
        """
        
        if data_format == 'bin':
            f = open(filename, 'rb')
            self.Ng = readbin(f, 'i')
            pos_i = f.tell()
            f.seek(0, 2)
            num_bytes = f.tell() - pos_i
            num_rows = num_bytes // 8 // (self.Ng * 2 + 5)
            if num_rows * 8 * (self.Ng * 2 + 5) != num_bytes:
                raise IOError(f"The number of bytes seems off: Ng={self.Ng}, num_bytes={num_bytes}")
            f.seek(4, 0)
            data = loadbin(f, num=num_bytes // 8).reshape([num_rows, self.Ng * 2 + 5])
            f.close()

            self.Ng += 2  # !! use Ng+2 in fact
            self.m = data[0][1:self.Ng + 1]
            self.a = data[0][self.Ng + 1:]
            self.t = data[1:, 0]
            self.dlog_m = np.diff(np.log10(self.m)).mean()
            self.dlog_a = np.diff(np.log10(self.a)).mean()
            self.sigma = data[1:, 1:self.Ng + 1]
            self.Nk = data[1:, self.Ng + 1:]
            self.Na = self.Nk * 3;
            self.dNda = self.Nk * 3 * self.a
            self.peak_Na = self.a[np.argmax(self.Na, axis=1)]
            self.peak_dNda = self.a[np.argmax(self.dNda, axis=1)]
            self.peak_sigma = self.a[np.argmax(self.sigma, axis=1)]

        elif data_format == 'ascii':
            data = np.loadtxt(filename)
            self.Ng = (data[0].size - 1) // 2
            self.m = data[0][1:self.Ng + 1]
            self.a = data[0][self.Ng + 1:]
            self.t = data[1:, 0]
            self.dlog_m = np.diff(np.log10(self.m)).mean()
            self.dlog_a = np.diff(np.log10(self.a)).mean()
            self.sigma = data[1:, 1:self.Ng + 1]
            self.Nk = data[1:, self.Ng + 1:]
            self.Na = self.Nk * 3
            self.dNda = self.Nk * 3 * self.a
            self.peak_Na = self.a[np.argmax(self.Na, axis=1)]
            self.peak_dNda = self.a[np.argmax(self.dNda, axis=1)]
            self.peak_sigma = self.a[np.argmax(self.sigma, axis=1)]

    def plot_peak_pos(self):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(self.t, self.peak_Na, lw=3, alpha=0.75, label='numerical (dN/dlna)')
        ax.loglog(self.t, self.peak_dNda, lw=3, alpha=0.75, label='numerical (dN/da)')
        ax.loglog(self.t, self.peak_sigma, lw=3, alpha=0.75, label='numerical (sigma)')
        ax.set(xlabel=r't [yr]', ylabel='peak position [cm]',
               title=f'total grids: {self.Ng - 2}; per mass decade: {1 / self.dlog_m:.3f}');
        return fig, ax



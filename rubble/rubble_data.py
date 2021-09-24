import traceback
import struct
from array import array
from numbers import Number
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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

    def __init__(self, filename, data_format='bin',
                 chop=True, chop_thresh=1e-200):
        """
        Read in simulation data for further analyses
        
        Parameters
        ----------
        filename : str
            name of the data file
        data_format : str
            'bin': the data is in binary format (by default)
            'ascii': the data is in ascii data
        chop : bool
            Replace tiny numbers by exact zeros (default=True)
        chop_thresh : float
            choose the threshold for chopping (default=1e-200,
            i.e., any number below 1e-200 will be replaced by 0)
        """
        
        if data_format == 'bin':
            f = open(filename, 'rb')
            self.Ng = readbin(f, 'i')
            pos_i = f.tell()
            f.seek(0, 2)
            num_bytes = f.tell() - pos_i
            num_rows = num_bytes // 8 // (self.Ng * 2 + 5)
            if num_rows * 8 * (self.Ng * 2 + 5) > num_bytes:
                raise IOError(f"The number of bytes seems off: Ng={self.Ng}, "
                              + f"num_rows={num_rows}, num_bytes={num_bytes}")
            elif num_rows * 8 * (self.Ng * 2 + 5) < num_bytes:
                print(f"Bytes more than data: Ng={self.Ng}, num_rows={num_rows}, num_bytes={num_bytes}"
                      f"\nreading anyway")
                num_rows -= 1
            else:
                pass

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
            if chop:
                chop_idx = self.sigma < chop_thresh
                self.Nk[chop_idx] = 0
                self.sigma[chop_idx] = 0
            self.Na = self.Nk * 3
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
            if chop:
                chop_idx = self.sigma < chop_thresh
                self.Nk[chop_idx] = 0
                self.sigma[chop_idx] = 0
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
               title=f'total grids: {self.Ng - 2}; per mass decade: {1 / self.dlog_m:.3f}')
        return fig, ax

    def add_mass_axis(self, ax, offset=-0.175):
        """ Add a mass axis below the x-axis which illustrate size """
        
        tmp_ax = ax.twiny(); tmp_ax.xaxis.set_ticks_position("bottom")
        tmp_ax.xaxis.set_label_position("bottom")
        tmp_ax.spines["bottom"].set_position(("axes", offset))
        tmp_ax.set_frame_on(True)
        tmp_ax.patch.set_visible(False)
        for sp in tmp_ax.spines.values():
            sp.set_visible(False)
        tmp_ax.spines["bottom"].set_visible(True)
        tmp_ax.loglog(self.m, self.sigma[0], alpha=0.0)
        tmp_ax.set(xlabel=r"m [g]")

    def plot_snapshots(self, selected_time, ax=None, **kwargs):
        """
        Plot snapshots of solid size distribution at selected time points

        Parameters
        ----------
        selected_time: list or np.ndarray
            an array of selected time to plot snapshot size distribution
        ax : axes object from matplotlib
            if not provided, this function will create a/an figure/axes object and return them
        kwargs
            label_fmt can be directly specified and will be passed to ax.loglog for plotting (default: "t={:.0f}yr")
            other keywords can be grouped in the following dictionaries:
                "plot_kw" will be passed to ax.loglog
                "ax_set_kw" will be passed to ax.set
        Returns
        -------
        both the figure and axes object if ax is not provided
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        else:
            fig = None
        for t_sel in selected_time:
            t_idx = np.argmin(abs(self.t - t_sel))
            ax.loglog(self.a, self.sigma[t_idx], lw=3, alpha=0.5,
                      label=kwargs.get("label_fmt", "t={:.0f}yr").format(self.t[t_idx]),
                      **(kwargs.get("plot_kw", {})))
        ax.set(xlabel=r'a [cm]', ylabel=r"$\sigma$ [g/cm$^2$]", **(kwargs.get("ax_set_kw", {})))
        ax.legend(loc='best', fontsize=14)
        self.add_mass_axis(ax)

        if fig is None:
            return None
        else:
            fig.tight_layout()
            return fig, ax

    def discretize_colormap(self, N, base_cmap='viridis',
                            cut_top=False, curve_tuning=1.0):
        """
        Create an N-bin discrete colormap from the specified input map

        Parameters
        ----------
        N : int
            how many colors in the desired colormap
        base_cmap : str
            the base colormap to discretize
        cut_top : bool
            whether or not to cut the original top color (for vmax)
            e.g., useful for removing the white color when base_cmap='hot'
        """

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N + 1))
        value_list = np.linspace(0, 1, N + 1) ** curve_tuning
        cmap_name = 'discretize ' + base.name
        if cut_top:
            return LinearSegmentedColormap.from_list(cmap_name, list(zip(value_list[:-1], color_list[:-1])), N)
        else:
            return LinearSegmentedColormap.from_list(cmap_name, list(zip(value_list, color_list)), N)

    def plot_time_evolution(self, t_range=None, num_t2plot=1024,
                            ax=None, cmap='viridis', num_colors=20, **kwargs):
        """
        Plot the time evolution of sigma on a colormap in log_a vs. log_t frame

        Parameters
        ----------
        t_range : list
            select a range of time interval to show, default: None
        num_t2plot : int
            number of snapshots to plot (vertical resolution of colormap), default: 1024
        ax : axes object from matplotlib
            if not provided, this function will create a/an figure/axes object and return them
        cmap : str
            colormap, default: 'viridis'
        num_colors : int
            number of discretized colors to plot, default: 20
            if set to None, then a smooth colormap will be used
        kwargs
            vmin and vmax can be directly specified and will be passed to ax.pcolor[fast]
            other parameters can be grouped in a dict named pcolor_kw and will also be passed
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        else:
            fig = None

        if t_range is None:
            log_t = np.logspace(0, np.log10(self.t.max()), num_t2plot)
        else:
            if isinstance(t_range, Number):
                print("Warning: t_range only has one value. Use it as the starting time")
                if t_range > self.t.max():
                    raise ValueError("t_range only has one value and exceeds the max time in data.")
                if t_range < 0:
                    t_range = 1
                log_t = np.logspace(np.log10(t_range), np.log10(self.t.max()), num_t2plot)
            elif isinstance(t_range, (tuple, list, np.ndarray)):
                t_range = np.asarray(t_range).flatten()
                if len(t_range) == 0:
                    print("Warning: no valid values in t_range. Use default settings.")
                elif len(t_range) == 1:
                    print("Warning: t_range only has one value. Use it as the starting time")
                    if t_range > self.t.max():
                        raise ValueError("t_range only has one value and exceeds the max time in data.")
                    if t_range < 0:
                        t_range = 1
                    log_t = np.logspace(np.log10(t_range), np.log10(self.t.max()), num_t2plot)
                else:
                    if t_range[0] > t_range[1]:
                        t_range[0], t_range[1] = t_range[1], t_range[0]
                    log_t = np.logspace(np.log10(max(1, t_range[0])),
                                        np.log10(min(self.t.max(), t_range[1])), num_t2plot)
            else:
                try:
                    if t_range[0] > t_range[1]:
                        t_range[0], t_range[1] = t_range[1], t_range[0]
                    log_t = np.logspace(np.log10(max(1, t_range[0])),
                                        np.log10(min(self.t.max(), t_range[1])), num_t2plot)
                except Exception as e:
                    print("Warning: there seems to be no valid values in t_range. Use default settings.")
                    log_t = np.logspace(0, np.log10(self.t.max()), num_t2plot)

        sigma_logyr = self.sigma[np.array([np.argmax(self.t >= tmp_t) for tmp_t in log_t])]
        try:
            # shading='auto' requires Matplotlib > 3.3
            ax.pcolor(np.log10(self.a), np.log10(log_t), np.log10(sigma_logyr), shading='auto',
                      vmin = kwargs.get("vmin", -17), vmax = kwargs.get("vmax", 3),
                      cmap = self.discretize_colormap(num_colors, base_cmap=cmap,
                                                      curve_tuning=kwargs.get("curve_tuning", 1.0)),
                      **(kwargs.get("pcolor_kw", {})))
        except Exception as e:
            ax.pcolorfast(np.log10(self.a), np.log10(log_t), np.log10(sigma_logyr),
                          vmin=kwargs.get("vmin", -17), vmax=kwargs.get("vmax", 3),
                          cmap=self.discretize_colormap(20, base_cmap=cmap,
                                                        curve_tuning=kwargs.get("curve_tuning", 1.0)),
                          **(kwargs.get("pcolor_kw", {})))

        ax.set(xlabel=r"log a [cm]", ylabel=r"log t [yr]")
        if fig is None:
            return None
        else:
            fig.tight_layout()
            return fig, ax

    def shrink_data(self, filename, sampling_rate,
                    keep_first_n=0):
        """ reduce data size with fewer records and output to a new file """

        self.new_dat_file = open(filename, 'wb')
        self.new_dat_file.write((self.Ng-2).to_bytes(4, 'little'))
        self.res4out = np.hstack([self.dlog_a, self.m, self.a])
        self.new_dat_file.write(self.res4out.tobytes())

        for i, t in enumerate(self.t):
            if i <= keep_first_n or i % sampling_rate == 0:
                self.res4out = np.hstack([t, self.sigma[i], self.Nk[i]])
                self.new_dat_file.write(self.res4out.tobytes())

        self.new_dat_file.close()



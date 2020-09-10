"""
Object to perform analysis and plotting on a given dataset
Methods for the measurement control software to anaylse and plot data
@author: krolljg
"""

import matplotlib.pyplot as plt
import numpy as np

import colorsys

import qcodes
#from qcodes import Instrument # consider making a qcodes instrument in the future - not sure what the advantage is

import qtt
from qtt.utilities.tools import addPPTslide


class MeasurementAnalysis():

    """
    Class that allows for analysis of measurement datasets. Must be initialised with a dataset for analysis.

    dataset: target dataset
    add_ppts: automatically loads plots into a powerpoint
    prev_fig: can initialise with a figure in order to continue working on it
    """
    def __init__(
            self,
            dataset: str,
            add_ppts: bool = True,
            prev_fig=None,
            verbose: bool = True,
            **kwargs
    ):
        #super().__init__(dataset+'Analysis', **kwargs)
        self.add_ppts = add_ppts
        self.load_data(dataset)
        if len(self.setpoint_vars) == 1:
            self.plot_1D()
        if len(self.setpoint_vars) == 2:
            self.plot_2D()

        # used to keep working on the same figure if necessary
        if prev_fig is None:
            self.init_fig()
        else:
            self.fig = prev_fig

    def load_data(self,dataset, xvar=None, yvar=None, zvar=None):
        self.dataset = dataset

        arrays = self.dataset.arrays

        self.setpoint_vars = {key:value for (key,value) in arrays.items() if arrays.get(key).is_setpoint}
        self.measured_vars = {key:value for (key,value) in arrays.items() if arrays.get(key).is_setpoint==False}

        # determine dimensionality of dataset, load x, y and z variables appropriately
        if len(self.setpoint_vars) == 1:
            if xvar is None:
                self.xvar = self.setpoint_vars.get(list(self.setpoint_vars)[0])
            else:
                self.xvar = self.setpoint_vars.get(xvar)

            if yvar is None:
                self.yvar = self.measured_vars.get(list(self.measured_vars)[0])
            else:
                self.yvar = self.measured_vars.get(yvar)
        else:
            if xvar is None:
                self.xvar = self.setpoint_vars.get(list(self.setpoint_vars)[0])
            else:
                self.xvar = self.setpoint_vars.get(xvar)

            if yvar is None:
                self.yvar = self.setpoint_vars.get(list(self.setpoint_vars)[1])
            else:
                self.yvar = self.setpoint_vars.get(yvar)

            if zvar is None:
                self.zvar = self.measured_vars.get(list(self.measured_vars)[0])
            else:
                self.zvar = self.measured_vars.get(zvar)



    def init_fig(self):
        ''' Initailised a new figure '''
        self.fig = plt.figure()

    def init_labels(self):
        ''' Used to generate a figure for 1D plots with axis labels and a title'''
        ax = self.fig.add_subplot(111)

        xvarlabel = self.xvar.label
        xvarunit = self.xvar.unit

        yvarlabel = self.yvar.label
        yvarunit = self.yvar.unit

        ax.set_xlabel(xvarlabel + ' (' + xvarunit + ')', fontsize=12)
        ax.set_ylabel(yvarlabel + ' (' + yvarunit + ')', fontsize=12)
        ax.set_title(str(self.dataset.location))

        ax.ticklabel_format(style='sci', scilimits=(0, 0))
        self.fig.tight_layout()

    def add_linetrace(self, dataset=None, xvar=None, yvar=None,  sub_fig=0, **kwargs):
        ''' Add linetrace to an existing figure '''
        if dataset is not None: # reloads data if new dataset
            self.load_data(dataset, xvar=xvar, yvar=yvar)
        ax = self.fig.axes[sub_fig] #can addres individual sub figures
        ax.plot(self.xvar, self.yvar, **kwargs)

    def extract_gates(self):
        ''' Extract the gate values from the metadata '''
        instruments = self.dataset.metadata.get('station').get('instruments')

        instrument_list = list(instruments.keys())
        ivvis = [inst for inst in instrument_list if inst[:4] == 'ivvi']
        dac_list = []
        dac_values = []
        for ivvi in ivvis:
            dac_list += instruments.get(ivvi).get('parameters')
            dacs = [dac for dac in dac_list if dac[:3] == 'dac']
            dac_values += [instruments.get(ivvi).get('parameters').get(dd).get('value') for dd in dacs]
        return dict(zip(dacs, dac_values))  # zip list toogether

    def add_ppt_slide(self,title=None,**kwargs):
        ''' Adds figure to a PPT, creates one of one is not open. '''
        gatelist = self.extract_gates()

        if title==None:
            if __name__ == '__main__':
                title = str(self.dataset.location)

        addPPTslide(fig=self.fig.number,title=title,notes=str(gatelist),**kwargs)

    def plot_1D(self, dataset=None, xvar=None, yvar=None, **kwargs):
        ''' Generates a 1D plot from a dataset. x and y can be specified by name.'''
        if dataset is not None:
            self.load_data(dataset,xvar,yvar)
        self.init_fig()


        self.init_labels()
        self.add_linetrace(**kwargs)
        if self.add_ppts:
            self.add_ppt_slide()

    def plot_2D(self, dataset=None, xvar=None, yvar=None, zvar=None, **kwargs):
        ''' Generates a 2D plot from a dataset. x y and z variables can be specified by name.'''
        if dataset is not None:
            self.load_data(dataset, xvar, yvar, zvar)
        self.init_fig()

        self.init_labels()

        cb = self.fig.axes[0].pcolormesh(self.xvar, self.yvar, self.zvar)
        self.fig.colorbar(cb)

        if self.add_ppts:
            self.add_ppt_slide()

    def calculate_resistance(self,dataset):
        self.plot_1D(dataset)
        # in future, add routine to calculate rescaling due to axes units (mV->V etc)

        fit = np.polyfit(self.xvar, self.yvar, 1)
        x_fit = np.linspace(self.xvar[0], self.xvar[-1], 100)
        y_fit = fit[0] * x_fit + fit[1]
        G = fit[0]
        R = (1 / G)

        self.fig.axes[0].plot(x_fit,y_fit,'k--',label = 'Resistance: %d Ohm'%R)
        self.fig.axes[0].legend()

        if self.add_ppts:
            self.add_ppt_slide()

    def determine_turn_on(self, threshold_factor=0.1, step=3):
        self.plot_1D()

        x = self.xvar
        y = self.yvar

        # check sweep direction and fix
        if y[0] > y[-1]:
            y = np.flip(y, 0)
            x = np.flip(x, 0)

        y_threshold = max(y) * threshold_factor

        # first position in y vector above threshold value:
        ind_start = np.argmax(np.asarray(y) > y_threshold)

        y_clean = y[ind_start:]
        x_clean = x[ind_start:]

        diff_vector = y_clean[step:] - y_clean[:-step]

        ind_diff_max = np.argmax(diff_vector)
        diff_max_y = max(diff_vector)
        diff_x = x_clean[ind_diff_max + step] - x_clean[ind_diff_max]
        slope = diff_max_y / diff_x

        pos_x = (x_clean[ind_diff_max + step] + x_clean[ind_diff_max]) / 2
        pos_y = (y_clean[ind_diff_max + step] + y_clean[ind_diff_max]) / 2

        offset_y = pos_y - pos_x * slope
        turn_on_value = int(np.round(-offset_y / slope, 0))

        y_fit = slope * np.asarray(x) + offset_y
        self.fig.axes[0].plot(x,y_fit,'k--',label = 'Turn on: %d mV'%turn_on_value)
        self.fig.axes[0].legend()
        self.fig.axes[0].set_ylim(bottom=min(y),top=max(y))

    def plot_drift_scans(self, forward_datasets, backward_datasets, xvar=None, yvar=None):
        self.load_data(forward_datasets[0], xvar, yvar)

        self.init_fig()

        self.init_labels()

        # generating my own colormap
        saturation = 0.8
        lightness = 0.8
        hue_range = np.linspace(0.0, 0.1, len(forward_datasets))
        color_list = [colorsys.hsv_to_rgb(hv, saturation, lightness) for hv in hue_range]

        for custom_color, fd, bd in zip(color_list, forward_datasets, backward_datasets):

            if custom_color == color_list[0]:
                self.add_linetrace(dataset=fd, xvar=xvar, yvar=yvar, color=custom_color, label='Forward')
                self.add_linetrace(dataset=bd, xvar=xvar, yvar=yvar, color=custom_color, linestyle='--', label='Backward')
            else:
                self.add_linetrace(dataset=fd, xvar=xvar, yvar=yvar, color=custom_color)
                self.add_linetrace(dataset=bd, xvar=xvar, yvar=yvar, color=custom_color, linestyle='--')
        self.fig.axes[0].legend()

        if self.add_ppts:
            self.add_ppt_slide()

    def analyse_drift_scan(self, forward_datasets, backward_datasets):
        # Written by Lucas (I think). Adapted with minimal changes.
        def scans_diff(x1, y1, x2, y2):  # ds1 should be shorter than ds2
            # check
            if len(x1) > len(x2):
                print('Error: cannot process datasets in reversed order')

            # sort both vectors in ascending order
            if y1[0] > y1[-1]:
                y1 = np.flip(y1, 0)
                x1 = np.flip(x1, 0)
            if y2[0] > y2[-1]:
                y2 = np.flip(y2, 0)
                x2 = np.flip(x2, 0)

            # Only select comparable part
            x2_trim = x2[:len(x1)]
            y2_trim = y2[:len(x1)]

            # check
            if max(abs(x1 - x2_trim)) > 0.001:
                print('Gate voltages are not comparable')
                print(x1)
                print(x2_trim)
                for i in [1]:
                    break

            # calculate sum of difference squared between both vectors
            y1_np = np.array(y1)
            y2_trim_np = np.array(y2_trim)
            try:
                y_diff_sq = sum((y1_np - y2_trim_np) ** 2)
            except:
                print('Error in calculating difference between two consecutive datasets')
            if (y_diff_sq / len(x1)) ** 0.5 < 0:
                print('ERROR: difference between datasets smaller than zero while it should be larger')
            return (y_diff_sq / len(x1)) ** 0.5

        ##############################################################################

        forward_diff_list = []
        backward_diff_list = []
        peak_voltage_list = []

        for i in range(len(forward_datasets) - 1):
            # FORWARD
            ds1 = forward_datasets[i]
            ds2 = forward_datasets[i + 1]

            self.load_data(ds1)
            x1, y1 = self.xvar, self.yvar
            self.load_data(ds2)
            x2, y2 = self.xvar, self.yvar

            rms_diff_FW = scans_diff(x1, y1, x2, y2)
            forward_diff_list.append(rms_diff_FW)

            # BACKWARD
            ds1 = backward_datasets[i]
            ds2 = backward_datasets[i + 1]

            self.load_data(ds1)
            x1, y1 = self.xvar, self.yvar
            self.load_data(ds2)
            x2, y2 = self.xvar, self.yvar

            rms_diff_BW = scans_diff(x1, y1, x2, y2)
            backward_diff_list.append(rms_diff_BW)

            # PEAK VOLTAGE LIST
            peak_voltage = max(x2)
            peak_voltage_list.append(peak_voltage)

        self.init_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(peak_voltage_list, forward_diff_list, '1r', label='Forward scans')
        ax.plot(peak_voltage_list, backward_diff_list, '2b', label='Backward scans')
        #        plt.yscale("log") #log scale
        plt.ylim(bottom=0)
        x_title1 = self.xvar.label
        plt.xlabel('Peak voltage on %s (mV)' % x_title1)
        plt.ylabel('RMS difference (A)')
        plt.legend()

        plt.tight_layout()
        if self.add_ppts:
            self.add_ppt_slide(title='RMS difference of drift scan')

        #return forward_diff_list, backward_diff_list
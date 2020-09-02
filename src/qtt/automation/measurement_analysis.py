"""
Object to perform analysis and plotting on a given dataset
Methods for the measurement control software to anaylse and plot data
@author: krolljg
"""

import matplotlib.pyplot as plt
import numpy as np

import qcodes
#from qcodes import Instrument # consider making a qcodes instrument in the future - not sure what the advantage is

class MeasurementAnalysis():

    """
    Class that allows for analysis of measurement datasets.
    """
    def __init__(
            self,
            dataset: str,
            verbose: bool = True,
            **kwargs
    ):
        #super().__init__(dataset+'Analysis', **kwargs)
        self.load_data(dataset)
        self.init_fig()

    def load_data(self,dataset):
        self.dataset = dataset
        arrays = self.dataset.arrays
        if len(arrays) > 2:
            Exception('plot_1D currently does not support datasets with >2 arrays')
        varnames = [k for k in arrays.keys()]
        is_setpoint_array = [arrays.get(a).is_setpoint for a in arrays]
        xvarpos = np.where(is_setpoint_array)[0][0]  # Extracting x pos in keys

        self.xvar = arrays.get(varnames[xvarpos])

        yvarpos = 1 - xvarpos
        self.yvar = arrays.get(varnames[yvarpos])


    def init_fig(self):
        self.fig = plt.figure()

    # could turn this set into an object, allowing it to be instantiated and kepe interal data like the arrays on fig
    def plot_1D(self):

        ax = self.fig.add_subplot(111)

        ax.plot(self.xvar, self.yvar)

        xvarlabel = self.xvar.label
        xvarunit = self.xvar.unit

        yvarlabel = self.yvar.label
        yvarunit = self.yvar.unit

        ax.set_xlabel(xvarlabel + ' (' + xvarunit + ')', fontsize=12)
        ax.set_ylabel(yvarlabel + ' (' + yvarunit + ')', fontsize=12)
        ax.set_title(str(self.dataset.location))

        ax.ticklabel_format(style='sci', scilimits=(0, 0))
        self.fig.tight_layout()

    def calculate_resistance(self):
        self.plot_1D()
        # in future, add routine to calculate rescaling due to axes units (mV->V etc)

        fit = np.polyfit(self.xvar, self.yvar, 1)
        x_fit = np.linspace(self.xvar[0], self.xvar[-1], 100)
        y_fit = fit[0] * x_fit + fit[1]
        G = fit[0]
        R = (1 / G)

        self.fig.axes[0].plot(x_fit,y_fit,'k--',label = 'Resistance: %d Ohm'%R)
        self.fig.axes[0].legend()






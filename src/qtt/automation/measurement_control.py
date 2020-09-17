# -*- coding: utf-8 -*-

"""
Created on Tue Sep  1 16:32:55 2020

IN DEVELOPMENT

atm - automated test measurements

Utility toolset that will eventually enable automated measurement of test structure devices.


Consists of a few important classes:

MeasurementControl - performs and analyses measurements according to a process
MeasurementAnalysis - used to analyse measurement datasets
MeasurementProcess - an algorithm that specifies the sequence of measurements to be performed
MeasurementMap - contains functions to work out correct gates and instruments for a given measurement
MeasurementGeometry - physical geometry of the sample, for now we will only consider linear arrays

@author: krolljg
"""
import qcodes
from qcodes import Instrument

import qtt
from qtt.measurements.scans import scanjob_t, scan1D, scan2D, scan1Dfeedback
#from qtt.automation.measurement_analysis import MeasurementAnalysis

import time
import numpy as np


class MeasurementControl(Instrument):

    """
    Class that allows for control of measurements.
    """

    def __init__(
            self,
            sample_name: str,
            station: object,
            datadir: str,
            autoanalysis: bool = True, #autoanalysis to be implemented
            liveplotting: bool = False,
            verbose: bool = True,
            **kwargs

    ):
        super().__init__(sample_name+'Control', **kwargs)

        qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
        self.station = station
        self.gates = station.gates
        self.autoanalysis = autoanalysis
        self.liveplotting = liveplotting
        self.verbose = verbose


    def scan_1D(self, scan_gate, start, end, step, meas_instr, pause_before_start=None, wait_time=0.02, abort_controller=None):
        ''' Used to sweep a gate and measure on some instruments '''
        if pause_before_start is not None:
            try:
                self.gates.set(scan_gate, start)
            except:
                scan_gate(start)
            time.sleep(pause_before_start)

        scanjob = scanjob_t({'sweepdata': dict({'param': scan_gate,
                                                'start': start,
                                                'end': end,
                                                'step': step,
                                                'wait_time': wait_time}), 'minstrument': meas_instr})

        if abort_controller is not None:
            dataset = scan1Dfeedback(self.station, scanjob, location=None, verbose=self.verbose, abort_controller=abort_controller)
        else:
            dataset = scan1D(self.station, scanjob, location=None, verbose=self.verbose)
        return dataset

    def scan_2D(self, sweep_gate, sweep_start, sweep_end, sweep_step, step_gate, step_start, step_end, step_step,
                meas_instr, pause_before_start=None, sweep_wait=0.02, step_wait=0.02):
        ''' Used to sweep a gate and measure on some instruments '''
        if pause_before_start is not None:
            try:
                self.gates.set(step_gate, step_start)
            except:
                step_gate(step_start)
            time.sleep(pause_before_start)
        scanjob = scanjob_t({'sweepdata': dict({'param': sweep_gate,
                                                'start': sweep_start,
                                                'end': sweep_end,
                                                'step': sweep_step,
                                                'wait_time': sweep_wait}),
                             'stepdata': dict({'param': step_gate,
                                               'start': step_start,
                                               'end': step_end,
                                               'step': step_step,
                                               'wait_time': step_wait}),
                             'minstrument': meas_instr})
        dataset = qtt.measurements.scans.scan2D(self.station, scanjob)


        return dataset


    def drift_scan(self, scan_gate, start, end_voltage_list, step, meas_instr, forward_datasets = None, backward_datasets= None):
        ''' Used to perform 1D sweeps up to increasingly higher voltages to look at drift '''

        try:
            self.gates.set(scan_gate, start)
        except:
            scan_gate(start)
        time.sleep(0.5)

        if forward_datasets is None:
            forward_datasets = []
        if backward_datasets is None:
            backward_datasets = []


        for end in end_voltage_list:
            dataset_forward = self.scan_1D(scan_gate, start, end, step, meas_instr)
            forward_datasets.append(dataset_forward)

            dataset_backward = self.scan_1D(scan_gate, end, start, step, meas_instr)
            backward_datasets.append(dataset_backward)

        return forward_datasets, backward_datasets

class FourProbeR(qcodes.Parameter):
    def __init__(self, name, Vparam, Iparam):
        super().__init__(name, label='Four probe resistance',
                         docstring='Calculates the four probe resistance from the last read V and I values.')
        self.Vparam = Vparam
        self.Iparam = Iparam
    # you must provide a get method, a set method, or both.
    def get_raw(self):
        V = self.Vparam.get_latest()
        I = self.Iparam.get_latest()

        if V or I is 0:
            raise Exception('V and I should be read out before reading R')

        R = V/I
        return R



# to make more generic would make it work with multiple
class DetermineTurnOn(Instrument):
    '''
    AbortController object to determine if a measurement should be stopped.

    Arguments:
    station - current measurement station
    meas_par - parameter we are monitoring
    threshold - parameter for the method that checks whether to abort
    **kwargs - passed to Instrument super init

    Method:
    check_abort(dataset)
    '''

    def __init__(self, station, **kwargs):
        super().__init__('Turn_On_Controller', **kwargs)
        self.station = station


    def set_params(self, meas_par, threshold):
        self.meas_par = meas_par
        self.threshold = threshold

    def check_abort(self, dataset):
        ''' Return True if the measurement should be aborted. '''
        return np.nanmax(dataset.arrays[self.meas_par.name]) > self.threshold



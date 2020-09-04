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
from qtt.measurements.scans import scanjob_t, scan1D, scan2D
#from qtt.automation.measurement_analysis import MeasurementAnalysis

import time



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


    def scan_1D(self, scan_gate, start, end, step, meas_instr, pause_before_start=None, wait_time=0.02):
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
        dataset = scan1D(self.station, scanjob, location=None, verbose=self.verbose)


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
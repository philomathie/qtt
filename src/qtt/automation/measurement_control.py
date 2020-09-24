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

import scipy.optimize as optimisation


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


class SoftSwitches():
    ''' Class to control softswitches to switch between measuring accumulation and constrictions.
    geometry - ordered list of gate and ohmic names representing their geometric layout '''

    def __init__(self, geometry=None, gates=None):
        if geometry == None:
            raise Exception('Please initialise with a geometry')
        self.geometry = geometry
        self.gates = gates
        self.ohmics = [oo for oo in geometry if oo[0] == 'O']

    def set_contacts(self, gate_name):
        ''' Pass string containing gate name, and the correct Ohmics will automatically be selected. '''

        for oo in self.ohmics:  # ensures other contacts are open
            self.gates.set(oo, 0)

        pos = self.geometry.index(gate_name)
        # turning on relevant ohmics
        self.gates.set(self.geometry[pos - 1], 2000)
        self.gates.set(self.geometry[pos + 1], 2000)

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

    def __init__(self, station, method = None, **kwargs):
        super().__init__('Turn_On_Controller', **kwargs)
        self.station = station
        self.method = method
        self.methodkwargs = {'threshold': 0.3e-9}
        self.ps = []

    def set_params(self, sweep_par, meas_par):
        self.sweep_par = sweep_par
        self.meas_par = meas_par

    def check_abort(self, dataset):
        ''' Return True if the measurement should be aborted. '''
        abort = False
        if self.method is None or self.method == 'threshold':
            threshold = self.methodkwargs['threshold']
            abort = np.nanmax(dataset.arrays[self.meas_par]) > threshold

        if self.method == 'gradient':
            abort = self.gradient_method(dataset)

        return abort



    def set_method(self,method, **kwargs):
        self.method = method
        self.methodkwargs = kwargs

    def gradient_method(self,dataset):

        abort = False

        def linearmodel(x, m, c):
            return x * m + c

        def linearfit(x, y):  # return fit and r squared value
            popt, pcov = optimisation.curve_fit(linearmodel, x, y, p0=[0, 0])

            residuals = y - linearmodel(x, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            return popt, r_squared


        y = dataset.arrays[self.meas_par]
        y = y[np.isfinite(y)]
        x = dataset.arrays[self.sweep_par][:len(y)]



        filterwindow = self.methodkwargs['filterwindow']
        gradient = self.methodkwargs['gradient']

        if len(x) >= filterwindow:
            xsub = x[-filterwindow:]
            ysub = y[-filterwindow:]
            popt, r_sq = linearfit(xsub, ysub)

            self.ps.append(popt)

            if popt[0] > gradient:
                abort = True
        else:
            abort = False
        return abort


class AutomatedMeasurement():
    '''
    Class to control automated measurements. Should initialise with station, device geometry and measurement method.
    station - qcodes station object
    geometry - list of gate names ordered by their physical location
    soft_switches - SoftSwitches object to set the right Ohmics to measure
    abort_controller - AbortController object used to determine when gates have turned on
    accs - list of accumulation gate names
    cons - list of constriction gate names
    meas_instr - list of instruments to measure with
    bias_voltage - default bias voltage to perform measurements at
    '''

    def __init__(self, measurement_controller, geometry, soft_switches, abort_controller,
                 accs, cons, meas_instr, bias_voltage=100e-6, step = 5, start = 0):
        self.MC = measurement_controller
        self.station = self.MC.station
        self.geometry = geometry
        self.AC = abort_controller
        self.soft_switches = soft_switches
        gate_datum = {'turn_on': None, 'hysteresis': None,'datasets': []}

        self.cons = cons
        self.accs = accs
        self.gate_data = {gg: gate_datum for gg in accs + cons}

        self.bias_voltage = bias_voltage
        self.meas_instr = meas_instr
        self.step = step
        self.start = start

    def measure_sample(self, safe_gate_voltage = 600, max_voltage = 1000, gate_increment = 100, con_offset = 200,
                       hysteresis_target = 1200):
        ''' Runs measurement procedure on sample. '''

        self.measure_turn_ons(self.accs, safe_gate_voltage, max_voltage, gate_increment)

        # sweep up constrictions to maximum of adjacent transistor turn on + constriction offset
        # if adjacent constrictions are not on, use lowest value for other accs then up to max voltage

        for cc in self.cons:
            con_target_voltage = max_voltage
            pos = self.geometry.index(cc)
            # turning on relevant ohmics
            self.soft_switches.set_contacts(cc)

            adjacent_accs = [self.geometry[pos - 2],self.geometry[pos + 2]]
            adjacent_acc_values = [self.gate_data[gg]['turn_on'] for gg in adjacent_accs if self.gate_data[gg]['turn_on'] is not None ]
            acc_values = [self.gate_data[gg]['turn_on'] for gg in self.accs if self.gate_data[gg]['turn_on'] is not None ]
            # extract lowest of 2 neighboring turn ons, alternatively all turn ons
            if adjacent_acc_values is not []:
                con_target_voltage = np.min(adjacent_acc_values) + con_offset
            elif acc_values is not []:
                con_target_voltage = np.min(acc_values) + con_offset

            self._measure_turn_on(cc, self.start, con_target_voltage)

        # once done, plot all the data

        for gg in self.gate_data:
            self.MC.plot_multiple_datasets(gg['datasets'])

        # finally perform hysteresis measurements
        for aa in self.accs:
            if self._check_gate_turn_on(aa) is not None:
                end_voltage_list = np.arange(self._check_gate_turn_on(aa)+con_offset,hysteresis_target,self.step*10)
                fwds, bwds = self.MC.drift_scan(aa, self.start, end_voltage_list, self.step, self.meas_instr)
                self.MA.plot_drift_scans(fwds,bwds)
                self.MA.analyse_drift_scans(fwds, bwds)

    def _check_gate_turn_on(self,gate):
        turn_on = self.gate_data[gate]['turn_on']
        return turn_on

    def _check_all_on(self,gate_list):
        """ Returns True if all gates in the gate_list have turned on """
        return [self.gate_data[gg]['turn_on'] is None for gg in gate_list].count(True) == 0


    def measure_turn_ons(self,gate_list, safe_gate_voltage = 600, max_voltage = 1000, gate_increment = 100):
        all_on = False
        sweep_target = safe_gate_voltage
        # work way through gate list, looking for turn on.
        while (sweep_target < max_voltage) and not all_on:
            for gg in gate_list:
                if self.gate_data[gg]['turn_on'] is None:
                    self.soft_switches.set_contacts(gg)
                    self._measure_turn_on(gg, self.start, sweep_target)

            sweep_target += gate_increment

            if self._check_all_on(gate_list):
                all_on = True  # stops procedure once all gates in gate_list are on.



    def _measure_turn_on(self, gate_name, start, end):

        # setting up the ohmics
        self.soft_switches(gate_name)
        scan_gate = self.gates[gate_name]

        # if gate is a constriction, turn off the constriction on the opposite side
        if gate_name[0]=='C':
            opp_gate_name = int(self.cons.index(gate_name)+ len(self.cons)/2)
            self.station.set(opp_gate_name,0)

        self.AC.set_params(gate_name, self.meas_instr[0].name)

        dataset = self.MC.scan_1D(scan_gate, start, end, self.step, self.meas_instr, abort_controller=self.AC)
        self.gate_data['gate_name']['datasets'].append(dataset)

        if self.AC.check_abort(dataset): # if measurement aborted, record turn on value
            self.gate_data['gate_name']['turn_on'] = scan_gate()
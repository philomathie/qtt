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
from qtt.automation.measurement_analysis import MeasurementAnalysis

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


    def scan_1D(self, scan_gate, start, end, step, meas_instr, pause_before_start=None, wait_time=0.02,
                abort_controller=None,plot_param=None,sub_plots=None):
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
            dataset = scan1Dfeedback(self.station, scanjob, location=None, verbose=self.verbose, abort_controller=abort_controller, plotparam=plot_param,subplots=sub_plots)
        else:
            dataset = scan1D(self.station, scanjob, location=None, verbose=self.verbose, plotparam=plot_param,subplots=sub_plots)
        return dataset

    def scan_2D(self, sweep_gate, sweep_start, sweep_end, sweep_step, step_gate, step_start, step_end, step_step,
                meas_instr, pause_before_start=None, sweep_wait=0.02, step_wait=0.02, plot_param=None):
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
        dataset = qtt.measurements.scans.scan2D(self.station, scanjob, plotparam=plot_param)


        return dataset


    def drift_scan(self, scan_gate, start, end_voltage_list, step, meas_instr, forward_datasets = None,
                   backward_datasets= None, auto_plot=False, threshold=None):
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

        MA = MeasurementAnalysis()

        for end in end_voltage_list:
            dataset_forward = self.scan_1D(scan_gate, start, end, step, meas_instr)
            forward_datasets.append(dataset_forward)

            dataset_backward = self.scan_1D(scan_gate, end, start, step, meas_instr)
            backward_datasets.append(dataset_backward)

            if auto_plot:
                MA.plot_multiple_scans(forward_datasets,backward_datasets)
                MA.plot_drift_scans(forward_datasets,backward_datasets)

            if threshold is not None:
                forward_max = np.max(MA.forward_diff_list)
                backward_max = np.max(MA.backward_diff_list)

                if (forward_max>threshold) or (backward_max>threshold):
                    break # stop the loop when we have entered hysteresis
        return forward_datasets, backward_datasets

    def find_hysteresis(self, scan_gate, start, end_voltage_list, step, meas_instr, plot_param=None, sub_plots=False, forward_datasets = None,
                   backward_datasets= None, threshold=None, pause_before_start=0):
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

        # creating analysis object for each figure. turning off powerpoint generation
        SweepAnalysis = MeasurementAnalysis(add_ppts=False)
        DriftAnalysis = MeasurementAnalysis(add_ppts=False)

        # creating empty hysteresis object
        hysteresis_point = None

        for end in end_voltage_list:
            dataset_forward = self.scan_1D(scan_gate, start, end, step, meas_instr, plot_param=plot_param, sub_plots=sub_plots, pause_before_start=pause_before_start)
            forward_datasets.append(dataset_forward)

            dataset_backward = self.scan_1D(scan_gate, end, start, step, meas_instr, plot_param=plot_param, sub_plots=sub_plots, pause_before_start=pause_before_start)
            backward_datasets.append(dataset_backward)


            SweepAnalysis.plot_drift_scans(forward_datasets,backward_datasets,new_fig=False)
            SweepAnalysis.fig.canvas.draw()
            DriftAnalysis.analyse_drift_scans(forward_datasets,backward_datasets,new_fig=False)
            DriftAnalysis.fig.canvas.draw()
            if (threshold is not None) and (len(DriftAnalysis.forward_diff_list)>=1):
                forward_max = np.max(DriftAnalysis.forward_diff_list)
                backward_max = np.max(DriftAnalysis.backward_diff_list)

                if (forward_max>threshold) or (backward_max>threshold):

                    # generate plots
                    SweepAnalysis.add_ppts = True
                    DriftAnalysis.add_ppts = True
                    SweepAnalysis.plot_drift_scans(forward_datasets, backward_datasets, new_fig=False)
                    DriftAnalysis.analyse_drift_scans(forward_datasets, backward_datasets, new_fig=False)
                    hysteresis_point = np.max(DriftAnalysis.xvar)
                    break  # stop the loop when we have entered hysteresis
        self.forward_datasets = forward_datasets
        self.backward_datasets = backward_datasets

        return forward_datasets, backward_datasets, hysteresis_point

class FourProbe(qcodes.Parameter):
    '''
    Qcodes metainstrument that measures four probe resistance or resistivity (for hallbars).
    name
    Vparam: qcodes parameter for voltage measurement
    Iparam: qcodes parameter for current measurement
    return_parameter='R': parameter to return. 'R', 'Rho_xx','Rho_xy'
    aspect_ratio=None: aspect ratio for hallbar used in Rho_xx
    I_threshold=1e-10: current threshold below which it returns 'nan'

    '''
    def __init__(self, name, Vparam, Iparam, return_parameter='R', aspect_ratio=None, I_threshold=1e-10):


        super().__init__(name, label=return_parameter, unit='Ohm')
        self.V_param = Vparam
        self.I_param = Iparam
        self.aspect_ratio = aspect_ratio
        self.I_threshold = I_threshold
        self.return_parameter = return_parameter

        if (return_parameter is 'rho_xx') and (aspect_ratio is None):
            raise Exception ('You must set the aspect ratio for rho measurements.')


    # you must provide a get method, a set method, or both.
    def get_raw(self):
        V = self.V_param.get()
        I = self.I_param.get()

        val = float('nan')

        if I > self.I_threshold:
            val = V/I

            if self.return_parameter.lower() == 'rho_xx':
                val = V/I/self.aspect_ratio

        return val


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
        self.gates.set(self.geometry[(pos + 1) %len(self.geometry)], 2000)
        time.sleep(1) # hard coded 1 second wait to allow switches to settle

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
    Class to control automated measurements. Should initialise with station, device geometry and measurement method. Bias voltage should be set beforehand.
    station - qcodes station object
    geometry - list of gate names ordered by their physical location
    soft_switches - SoftSwitches object to set the right Ohmics to measure
    abort_controller - AbortController object used to determine when gates have turned on
    accs - list of accumulation gate names
    cons - list of constriction gate names
    meas_instr - list of instruments to measure with
    '''

    def __init__(self,measurement_controller, measurement_analysis, geometry, soft_switches, abort_controller,
                 accs, cons, meas_instr, bias_voltage=100e-6, step = 5, start = 0, turn_on_ramp = 100, pause_before_start=2):
        self.MC = measurement_controller
        self.MA = measurement_analysis
        self.station = self.MC.station
        self.geometry = geometry
        self.AC = abort_controller
        self.soft_switches = soft_switches
        self.cons = cons
        self.accs = accs
        self.gate_data = {gg: {'turn_on': None,
                               'hysteresis': None,
                               'datasets': []
                               } for gg in accs + cons}

        self.meas_instr = meas_instr
        self.step = step
        self.start = start
        self.turn_on_ramp = turn_on_ramp
        self.p_b_s = pause_before_start
    def measure_sample(self, safe_gate_voltage = 600, max_voltage = 1000, gate_increment = 100, con_offset = 200,
                       hysteresis_target = 1200):
        ''' Runs measurement procedure on sample. '''

        self.measure_turn_ons(self.accs, safe_gate_voltage, max_voltage, gate_increment)

        # sweep up constrictions to maximum of adjacent transistor turn on + constriction offset
        # if adjacent constrictions are not on, use lowest value for other accs then up to max voltage
        
        # moving to sweep the gates that turned on by the 'turn on ramp' before measuring constriction 
        for aa in self.accs:
            if self.gate_data[aa]['turn_on'] is not None:
                self._measure_turn_on(aa, self.start, self.gate_data[aa]['turn_on']+self.turn_on_ramp, abortable=False)
        
        
        print('Moving to measure constrictions')
        for cc in self.cons:
            con_target_voltage = max_voltage
            pos = self.geometry.index(cc)
            # turning on relevant ohmics
            self.soft_switches.set_contacts(cc)

            adjacent_accs = [self.geometry[pos - 2],self.geometry[(pos + 2)%len(self.geometry)]]
            adjacent_acc_values = [self.gate_data[gg]['turn_on'] for gg in adjacent_accs if self.gate_data[gg]['turn_on'] is not None ]
            acc_values = [self.gate_data[gg]['turn_on'] for gg in self.accs if self.gate_data[gg]['turn_on'] is not None ]
            # extract lowest of 2 neighboring turn ons, alternatively all turn ons
            if adjacent_acc_values is not []:
                con_target_voltage = np.min(adjacent_acc_values) + con_offset + self.turn_on_ramp
                print('Lowest adjacent turn on is: '+str(np.min(adjacent_acc_values))+', increasing by: '+str(con_offset+ self.turn_on_ramp))
            elif acc_values is not []:
                con_target_voltage = np.min(acc_values) + con_offset + self.turn_on_ramp
                print('Lowest global turn on is: '+str(np.min(acc_values))+', increasing by: '+str(con_offset+ self.turn_on_ramp))
            print(str(cc) + str(self.start) + str(con_target_voltage))
            self._measure_turn_on(cc, self.start, con_target_voltage)

        # once done, plot all the data

        for gg in self.gate_data:
            self.MA.plot_multiple_scans(self.gate_data[gg]['datasets'])

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
                print ('Measuring gate '+gg)
                if self.gate_data[gg]['turn_on'] is None:
                    self.soft_switches.set_contacts(gg)
                    self._measure_turn_on(gg, self.start, sweep_target)
                else:
                    print ('Gate '+gg+' already measured, skipping.')

            sweep_target += gate_increment
            print ('Incrementing gate sweep. Current target: '+str(sweep_target))

            if self._check_all_on(gate_list):
                all_on = True  # stops procedure once all gates in gate_list are on.
                
       
        print('Finished measuring turn ons for gates:' +str(gate_list))


    def _measure_turn_on(self, gate_name, start, end, abortable=True):

        # setting up the ohmics
        self.soft_switches.set_contacts(gate_name)
        scan_gate = self.station.gates[gate_name]

        # if gate is a constriction, turn off the constriction on the opposite side
        if gate_name[0]=='C':
            opp_gate_name = self.cons[(int(self.cons.index(gate_name)+ len(self.cons)/2)%len(self.cons))]
            self.station.gates.set(opp_gate_name,0)

        self.AC.set_params(gate_name, self.meas_instr[0].name)
        if abortable:
            dataset = self.MC.scan_1D(scan_gate, start, end, self.step, self.meas_instr, abort_controller=self.AC, pause_before_start=self.p_b_s)
        else: #used to push gate to specific value
            dataset = self.MC.scan_1D(scan_gate, start, end, self.step, self.meas_instr, pause_before_start=self.p_b_s)
        self.gate_data[gate_name]['datasets'].append(dataset)

        if self.AC.check_abort(dataset): # if measurement aborted, record turn on value
            self.gate_data[gate_name]['turn_on'] = scan_gate()
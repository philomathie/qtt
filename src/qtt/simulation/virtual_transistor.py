"""
Virtual version of a simple 2DEG transistor to be measured via Vbias

The system consists of: 
    
    - a single transistor
    - a top gate
    
There are virtual instruents for:
    - DACs: 1 for the gate, 1 for the ohmic bias
    - A keithley to read out the measured current
    - A virtual gates object
    
"""


import logging
import threading
from functools import partial

import numpy as np
import qcodes
from qcodes import Instrument

import qtt
from qtt.instrument_drivers.gates import VirtualDAC
from qtt.instrument_drivers.virtual_instruments import VirtualMeter, VirtualIVVI

logger = logging.getLogger(__name__)


def gate_settle(gate):
    """ Return gate settle times """

    return 0  # the virtual gates have no latency


def gate_boundaries(gate_map):
    """ Return gate boundaries

    Args:
        gate_map (dict)
    Returns:
        gate_boundaries (dict)
    """
    gate_boundaries = {}
    for g in gate_map:
        gate_boundaries[g] = (-2000, 2000)
    return gate_boundaries

    
    
def generate_configuration(ntrans=1):
    """ Generate configuration for n transistors(to be extended)

    Args:
        ntrans (int): number of transistors
    Returns:
        number_dac_modules (int)
        gate_map (dict)
        gates (list)
    """
    gates = []
    for ii in range(1,ntrans+1):
        gates += ['Acc%d' % ii]
        gates += ['VBias%d' % ii]

    number_dac_modules = int(np.ceil(len(gates) / 14))
    gate_map = {}
    for ii, g in enumerate(sorted(gates)):
        i = int(np.floor(ii / 16))
        d = ii % 16
        gate_map[g] = (i, d + 1)

    return number_dac_modules, gate_map, gates
    


# Defining simple model with 2 gates, one for the ohmic and one for the plunger
class TransistorModel(Instrument):
    """
    Simulation model for a simple transistor, with one accumulation gate that is measured in voltage bias.
    """
    
    def __init__(self, name, VDeamp, IAmp, n_trans=1, INoise=0, measurement_lag=0,
                 hysteresis_point=10000, drift_scale=10, verbose=0,  **kwargs):
        """
        Args:
            name (str): name for the instrument
            
        """
        
        super().__init__(name, **kwargs)

        # initialising with n transistors gates for now.
        number_dac_modules, gate_map, gates = generate_configuration(n_trans)
        
        self.nr_ivvi = number_dac_modules
        self.gate_map = gate_map

        # dictionary to hold the data of the model
        self._data = dict()
        self.lock = threading.Lock()

        # make parameters for all the gates...
        gate_map = self.gate_map

        self.gates = list(gate_map.keys())

        self.gate_pinchoff = 400
        
        #setting V deamp and I amp settings
        self.VDeamp = VDeamp
        self.IAmp = IAmp


        self.INoise = INoise
        self.prev_gate = 0 # variables to record sweep direction
        self.sweep_direction = +1
        self.measurement_lag = measurement_lag
        self.hysteresis_point = hysteresis_point
        self.drift_scale = drift_scale
        self.in_hysteresis = False
        self.highest_gate_value = 0

        gateset = [(i, a) for a in range(1, 17) for i in range(number_dac_modules)]
        for i, idx in gateset:
            g = 'ivvi%d_dac%d' % (i + 1, idx)
            logging.debug('add gate %s' % g)
            self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=partial(self._data_get, g),
                               set_cmd=partial(self._data_set, g),
                               )

        # make entries for keithleys
        for instr in ['keithley1']:
            if not instr in self._data:
                self._data[instr] = dict()
            g = instr + '_amplitude'
            self.add_parameter(g,
                               label=f'Amplitude {g} (nA)',
                               get_cmd=partial(getattr(self, instr + '_get'), 'amplitude'),
                               )


    def get_idn(self):
        ''' Overrule because the default get_idn yields a warning '''
        IDN = {'vendor': 'QuTech', 'model': self.name,
               'serial': None, 'firmware': None}
        return IDN

    def _data_get(self, param):
        return self._data.get(param, 0)

    def _data_set(self, param, value):
        self._data[param] = value
        return

    def gate2ivvi(self, g):
        i, j = self.gate_map[g]
        return 'ivvi%d' % (i + 1), 'dac%d' % j

    def gate2ivvi_value(self, g):
        i, j = self.gate2ivvi(g)
        value = self._data.get(i + '_' + j, 0)
        return value

    def get_gate(self, g):
        return self.gate2ivvi_value(g)
    
    def _calculate_resistance(self, gate, offset, R_sat):
        """ Calculate resistance in Ohms due to pinchoff gates """
        G_sat= 1./R_sat
        v = self.gate2ivvi_value(gate) # gate and offset are both in mV
        G = G_sat * qtt.algorithms.functions.logistic(v, offset, 1 / 40.)
        R= 1./G
        return R
    
    def _calculate_current(self, gates, offset=400., R_sat=1e5):
        """ Calculate the mesaured current due to pinchoff and Ohmic bias """

        # Extracting the bias gate and acc gate, hard coded for now
        VBias = gates.index('VBias1')
        Acc = gates.index('Acc1')


        # calculating sweep direction
        current_gate = self.gate2ivvi_value(gates[Acc])

        if current_gate >= self.prev_gate:
            self.sweep_direction = +1
        else:
            self.sweep_direction = -1
        self.prev_gate = current_gate

        # abort measurement testing
        #if current_gate > 500:
        #   qtt.abort_measurements(1)

        # Setting highest gate value
        if current_gate > self.highest_gate_value:
            self.highest_gate_value = current_gate

        gate_drift = self.sweep_direction * self.measurement_lag
        if current_gate > self.hysteresis_point:
            self.in_hysteresis = True

        if self.in_hysteresis:
            gate_drift += np.exp((self.highest_gate_value-self.hysteresis_point)*self.drift_scale/(self.hysteresis_point)) - 1
        totaloffset = offset + gate_drift

        
        R = self._calculate_resistance(gates[Acc], offset=totaloffset,R_sat=R_sat)
        I = self.gate2ivvi_value(gates[VBias])*self.VDeamp/R
        
        if self.INoise:
                I = I + (np.random.rand()-0.5) * self.INoise


        return I

    def compute(self):
        """ Compute output of the model """
    
        try:
            # current through transistors
            val = self._calculate_current(self.gates, offset=self.gate_pinchoff)
    
            self._data['instrument_amplitude'] = val
    
        except Exception as ex:
            print(ex)
            logging.warning(ex)
            val = 0
        return val

    def keithley1_get(self, param):
        with self.lock:
            val = self.compute()*self.IAmp
            self._data['keithley1_amplitude'] = val
        return val

def close(verbose=1):
    """ Close all instruments """
    global station, model, _initialized

    if station is None:
        return

    for instr in station.components.keys():
        if verbose:
            print('close %s' % station.components[instr])
        try:
            station.components[instr].close()
        except BaseException:
            print('could not close instrument %s' % station.components[instr])

    _initialized = False

# creating station for qcodes

# %%
_initialized = False

# pointer to qcodes station
station = None

# pointer to model
model = None


def boundaries():
    global model
    if model is None:
        raise Exception('model has not been initialized yet')
    return gate_boundaries(model.gate_map)

# %%


def getStation():
    global station
    return station


def initialize(IAmp, VDeamp, reinit=False, n_trans=1, start_manager=False, INoise=0, measurement_lag=0,
                 hysteresis_point=10000, drift_scale = 1, verbose=2, ):

    global station, _initialized, model

    logger.info('virtualTransistor: start')
    if verbose >= 2:
        print('initialize: create virtual transistor system')
    print ("Value of initialized is"+str(_initialized))
    if _initialized:
        if reinit:
            close(verbose=verbose)
        else:
            return station
    logger.info('virtualTrans: make TransModel')
    model = TransistorModel(name=qtt.measurements.scans.instrumentName('transistormodel'),
                     verbose=verbose >= 3, n_trans=n_trans, INoise=INoise,IAmp=IAmp, VDeamp=VDeamp,
                            measurement_lag=measurement_lag, hysteresis_point=hysteresis_point, drift_scale = drift_scale)
    gate_map = model.gate_map
    if verbose >= 2:
        logger.info('initialize: TransistorModel created')
    ivvis = []
    for ii in range(model.nr_ivvi):
        ivvis.append(VirtualIVVI(name='ivvi%d' % (ii + 1), model=model))
    gates = VirtualDAC(name='gates', gate_map=gate_map, instruments=ivvis)
    gates.set_boundaries(gate_boundaries(gate_map))

    logger.info('initialize: set values on gates')
    for g in model.gates:
        gates.set(g, 0)


    logger.info('initialize: create virtual keithley instruments')
    keithley1 = VirtualMeter('keithley1', model=model)

    logger.info('initialize: create station')
    station = qcodes.Station(gates, keithley1, *ivvis,
                             model, update_snapshot=False)
    station.metadata['sample'] = 'virtual_transistor'
    station.model = model

    station.gate_settle = gate_settle
    station.depletiongate_name = 'Acc1'
    station.channel_current = station.keithley1.amplitude

    station.jobmanager = None
    station.calib_master = None

    _initialized = True
    if verbose:
        print('initialized virtual transistor system (%d dots)' % n_trans)
    return station

# %%

# not sure what this does :)
if __name__ == '__main__' and 1:
    np.set_printoptions(precision=2, suppress=True)

    try:
        close()
    except BaseException:
        pass

    station = initialize(reinit=True, verbose=2)
    self = station.model
    model = station.model
    model.compute()
    _ = station.keithley1.amplitude()
    np.set_printoptions(suppress=True)






"""
A GUI for multi-processing logging using ZMQ

Pieter Eendebak <pieter.eendebak@tno.nl>

"""

#%%
import logging
import os
import random
import sys
import time
import logging

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


import zmq
from zmq.log.handlers import PUBHandler


#import pmatlab
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', default=1, help="verbosity level")
parser.add_argument('-l', '--level', default=logging.DEBUG, help="logging level")
parser.add_argument('-p', '--port', type=int, default=5800, help="zmq port")
parser.add_argument('-g', '--gui', type=int, default=1, help="show gui")
args = parser.parse_args()


#%% Util functions


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds """
    if (time.time() - tprint.time) > dt:
        print(string)
        tprint.time = time.time()
        if output:
            return True
        else:
            return
    else:
        if output:
            return False
        else:
            return

#%% Functions for installing the logger


def removeZMQlogger(name=None):
    logger=logging.getLogger(name)
    
    for h in logger.handlers:
        if isinstance(h, zmq.log.handlers.PUBHandler):
            print('removing handler %s' % h)
            logger.removeHandler(h)
        
def installZMQlogger(port=5800, name=None, clear=True, level=logging.INFO):
    if clear:
        removeZMQlogger(name)
        
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
#    pub.setsockopt(zmq.LINGER, 100)   
    pub.setsockopt(zmq.RCVHWM, 10)    # ?
    #ZMQ::HWM
    
    pub.connect('tcp://127.0.0.1:%i' % port)

    if name is None:
        rootlogger = logging.getLogger()
    else:
        rootlogger = logging.getLogger(name)
    if level is not None:
        rootlogger.setLevel(level)
    handler = PUBHandler(pub)
    rootlogger.addHandler(handler)
    return rootlogger

#%%

def sub_logger(port, level=logging.INFO, verbose=1):
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind('tcp://127.0.0.1:%i' % port)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 10)   
    
    logging.basicConfig(level=level)

    print('connected to port %s' % port)
    while True:
        pmatlab.tprint('still logging...', dt=2)
        time.sleep(.05)
        try:
            level, message  = sub.recv_multipart(zmq.NOBLOCK)      
            #level, message = sub.recv_multipart()
            message = message.decode('ascii')
            if message.endswith('\n'):
                # trim trailing newline, which will get appended again
                message = message[:-1]
            level=level.lower().decode('ascii')
            log = getattr(logging, level)
            log(message)
            if verbose>=2:
                print('message: %s (level %s)' % (message, level))
        except:
            message=''
            level=None
            pass
        #print('message: %s (level %s)' % (message, level))

def log_worker(port, interval=1, level=logging.DEBUG):
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.connect('tcp://127.0.0.1:%i' % port)

    logger = logging.getLogger(str(os.getpid()))
    logger.setLevel(level)
    handler = PUBHandler(pub)
    logger.addHandler(handler)
    print("starting logger at %i with level=%s" % (os.getpid(), level))

    while True:
        level = random.choice( [logging.DEBUG, logging.INFO,
              logging.WARN, logging.ERROR, logging.CRITICAL])

        logger.log(level, "Hello from %i!" % os.getpid())
        time.sleep(interval)


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.pyqtSignal(str)
    def flush( self ):
        pass
    def fileno( self ):
        return -1
    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(str(msg))
    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout
    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr
        
class HorseLoggingGUI(QtWidgets.QDialog):
    
    LOG_LEVELS = dict({logging.DEBUG: 'debug', logging.INFO: 'info',
              logging.WARN: 'warning', logging.ERROR: 'error', logging.CRITICAL: 'critical'})
    
    
    def __init__( self, parent = None ):
        super(HorseLoggingGUI, self).__init__(parent)

        self.setWindowTitle('ZMQ logger')

        self.imap=dict((v, k) for k, v in self.LOG_LEVELS.items())

        #self._console = QtWidgets.QTextBrowser(self)
        self._console = QtWidgets.QPlainTextEdit(self)
        self._console.setMaximumBlockCount(2000)
        
        self._button  = QtWidgets.QPushButton(self)
        self._button.setText('Clear')
        self._levelBox  = QtWidgets.QComboBox(self)
        for k in sorted(self.LOG_LEVELS.keys()):
            #print('item %s' % k)
            val=self.LOG_LEVELS[k]            
            self._levelBox.insertItem(k, val)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self._button)
        blayout.addWidget(self._levelBox)
        self._button.clicked.connect(self.clearMessages)
        self._levelBox.currentIndexChanged.connect(self.setLevel)

        #HorseLoggingGUI.LOG_LEVELS
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._console)
        #layout.addWidget(self._button)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.addMessage('logging started...\n')
        
        self._levelBox.setCurrentIndex(1)

        self.loglevel=logging.INFO
        
    def setLevel(self, boxidx):
        name = self._levelBox.itemText(boxidx)
        lvl=self.imap.get(name, None)
        print('set level to %s: %d' % (name, lvl) )
        if lvl is not None:
            self.loglevel=lvl
            
    def addMessage(self, msg, level=None):
        #self._console.insertPlainText(msg) 
        #self._console.append(msg) 
    
        if level is not None:
            if level<self.loglevel:
                return
        self._console.moveCursor (QtGui.QTextCursor.End);
        self._console.insertPlainText (msg);
        self._console.moveCursor (QtGui.QTextCursor.End);

    def clearMessages( self ):
        dlg._console.clear()
        self.addMessage('cleared messages...\n')


def qt_logger(port, dlg, level=logging.INFO, verbose=1):
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind('tcp://127.0.0.1:%i' % port)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 10)   
    
    logging.basicConfig(level=level)
    app=QtWidgets.QApplication.instance()
    app.processEvents()

    print('connected to port %s' % port)
    loop=0
    while True:
        tprint('ZMQ logger: logging...', dt=5)
        loop=loop+1
        try:
            level, message  = sub.recv_multipart(zmq.NOBLOCK)      
            #level, message = sub.recv_multipart()
            message = message.decode('ascii')
            if message.endswith('\n'):
                # trim trailing newline, which will get appended again
                message = message[:-1]
            level=level.lower().decode('ascii')
            log = getattr(logging, level)
            lvlvalue=dlg.imap.get(level, None)
            
            log(message)
            dlg.addMessage(message+'\n', lvlvalue)
            
            app.processEvents()
            
            if verbose>=2:
                print('message: %s (level %s)' % (message, level))
        except:
            # no messages in system....
            app.processEvents()
            time.sleep(.06)
            message=''
            level=None
            pass
            if dlg.isHidden():
                # break if window is closed
                break
            if loop>100:
                pass
                #break
        
#%%
if __name__ == '__main__':

    port=args.port
    verbose=args.verbose
    
    app = None
    if ( not QtWidgets.QApplication.instance() ):
        app = QtWidgets.QApplication([])
    dlg = HorseLoggingGUI()
    dlg.resize( 800, 400)
    #dlg.setGeometry(10,110,800,400)
    dlg.setGeometry(-1900,40,800,500); app.processEvents() # V2
    dlg.show();  app.processEvents()

    # start the log watcher
    try:
        #sub_logger(port, level=args.level, verbose=verbose)
        qt_logger(port, level=args.level, verbose=verbose, dlg=dlg)
        pass
    except KeyboardInterrupt:
        print('keyboard interrupt' )
        pass

    #if ( app ):
    #    app.exec_()

#%% Send message to logger
if 0:
    port=5800
    import logging
    from qtt.loggingGUI import installZMQlogger
    installZMQlogger(port=port, level=logging.INFO)
    logging.warning('test')
    #log_worker(port=5700, interval=1)
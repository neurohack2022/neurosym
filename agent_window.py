import csv
import logging
import random
import sys
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes
from PyQt5 import Qt, QtCore, QtGui
from PyQt5.QtCore import QTimer  # Qt,
from PyQt5.QtGui import QBrush, QFont, QPainter, QPen, QPolygon
from PyQt5.QtOpenGL import *
from PyQt5.QtWidgets import *

from soundscape import soundscape

from Board import CONNECT, Board, get_board_id

import time
from agent import Conducter
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

class agent_win(QMainWindow):
        def __init__(
            self,
            hardware=None,
            model=None,
            sim_type=None,
            data_type=None,
            serial_port=None,
            save_file=None,
            parent=None,
            board_id=None,
            abData=None,
            # board=None,
            ):
            super().__init__()
            self.setMinimumSize(800,200)
        
            # setting window title
            self.setWindowTitle('Agent Window')
            self.parent = parent
            
            # init layout
            self.layout = QGridLayout()
            widget = QWidget()
            widget.setLayout(self.layout)
            self.setCentralWidget(widget)

            self.timer = QTimer()
            # self.timer.setSingleShot(True)
            # self.timer.timeout.connect(self.run_forever)
            # self.timer.start(50)
            self.timer.setInterval(2000)
            # here is where the stream is happening?
            self.timer.timeout.connect(self.update)
            self.timer.start()

        def update(self):
            # soundscape().main(self.parent.ab_hist)
            print(self.parent.ab_hist[0][49])

            # here is where you create your widgets and add them to the layout

        def closeEvent(self, event):
            # this code will autorun just before the window closes
            
            event.accept()

if __name__ == '__main__':    
    app = QApplication(sys.argv)    
    win = agent_win() 
    win.show() 
    app.exec()
    
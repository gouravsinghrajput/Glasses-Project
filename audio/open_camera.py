import numpy as np 
import time 
import pyaudio 


from openwakeword.model import Model 
from faster_whisper import WhisperModel 


RATE = 16000
CHUNK = 1280 


WAKE_THRESHOLD = 0.5 
COMMAND_TIME = 5 




import time 
import numpy as np 
import pyaudio 
import string 

from openwakeword.model import Model 
from faster_whisper import WhisperModel 


AUDIO_PROGRAM_CLOSING_WORDS = ['close the audio detection', 
                               'close the program' 
                               ' ']
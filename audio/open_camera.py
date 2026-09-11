import numpy as np 
import time 
import pyaudio


from openwakeword.model import Model 
from faster_whisper import WhisperModel 


RATE = 16000
CHUNK = 1280 
RECORD_TIME = 4


WAKE_THRESHOLD = 0.5 
COMMAND_TIME = 5 


COMMANDS = ["open camera", 
            "open the camera", 
            "can you please open the camera for me",
            "start the camera", 
            "turn on the camera", 
            "activate the camera", 
            "activate camera"
            ]



model = WhisperModel(
    "small", 
    device = 'cpu', 
    compute_type = 'int8'
)


audio_input = pyaudio.PyAudio()


stream = audio_input.open(
    format = pyaudio.paInt16, 
    channels = 1, 
    rate = RATE, 
    input = True, 
    frames_per_buffer = CHUNK
)

#this printing is just for the testing
print ('listning...')

frames = []


# the loop variable doesn't matter here
#the number of iteration doesn't matter here
#what matters is whether the complete thing is iterated or not


# the iteration will go from basically 0 to RATE / CHUNK * RECORD_TIME
#this is the number of frames that will be procesed in a sec
for i in range (int(RATE / CHUNK * RECORD_TIME)):
    
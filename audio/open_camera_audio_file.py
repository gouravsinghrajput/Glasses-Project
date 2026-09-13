import numpy as np 
import time 
import pyaudio
import string 


from openwakeword.model import Model 
from faster_whisper import WhisperModel 

COMMANDS = ["open camera", 
                "open the camera", 
                "can you please open the camera for me",
                "start the camera", 
                "turn on the camera", 
                "activate the camera", 
                "activate camera"
                ]


BREAK_WORD = ['close the camera', 
                  'close camera', 
                  ]



def open_camera_audio_call():

    RATE = 16000
    CHUNK = 1280 
    RECORD_TIME = 2


    WAKE_THRESHOLD = 0.5 
    COMMAND_TIME = 5 



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

        data_taken = stream.read(
            CHUNK, 
            exception_on_overflow = False
        )

        frames.append(data_taken)


    audio_data_taken = np.frombuffer(
        b"".join(frames), 
        dtype = np.int16
    )

    audio_data_taken = audio_data_taken.astype(np.float32) / 32768.0


    segments, info = model.transcribe(
        audio_data_taken,
        language = "en",
        beam_size = 1
        )


    spoken_text = " ".join(
        segment.text for segment in segments
    ).strip()


    spoken_text = spoken_text.translate(str.maketrans("", "", string.punctuation))

    spoken_text = spoken_text.lower()

    #for testing purpose
    print("You said:", spoken_text)
    # print("Expected:", COMMANDS)


    #for testing purposes
    # if spoken_text in [command.lower() for command in COMMANDS]:

    #     print("✅ Correct!")

    # else:

    #     print("❌ Incorrect!")


    stream.stop_stream()
    stream.close()
    audio_input.terminate()

    return spoken_text
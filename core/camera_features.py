from audio.open_camera import open_camera_audio_call, COMMANDS, BREAK_WORD
from vision.camera_opening import camera_opening


while True:

    spoken_text = open_camera_audio_call() 

    if spoken_text in [command.lower() for command in COMMANDS]:

        camera_opening() 

    else:

        print ("retry")

    if spoken_text in [break_word.lower() for break_word in BREAK_WORD]:

        break
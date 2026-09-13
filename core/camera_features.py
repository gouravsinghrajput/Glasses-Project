from audio.open_camera import open_camera_audio_call, COMMANDS, BREAK_WORD
from vision.camera_opening import camera_opening
import cv2 as cv 



while True:

    spoken_text  = open_camera_audio_call()

    if spoken_text.lower() in [command.lower() for command in COMMANDS]:

        camera_opening() = True 

        while True:

            spoken_text = open_camera_audio_call()

            cap = camera_opening()

            if spoken_text.lower() in [break_word.lower() for break_word in BREAK_WORD]:


                cap.release()    
                cv.destroyAllWindows()


    else:

        print('retry')          
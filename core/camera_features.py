from audio.open_camera import open_camera_audio_call 
from vision.camera_opening import camera_opening


if open_camera_audio_call.spoken_text in [command.lower() for command in open_camera_audio_call.COMMANDS]:
    camera_opening()

else:
    print('retry')
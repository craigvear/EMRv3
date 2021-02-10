from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play


class SoundBot(): # smooths the data as a thread class
    def __init__(self):
        # audio source variables
        audio_file = 'data/misha_lacy_off_minor.wav'
        self.audio = AudioSegment.from_wav(audio_file)
        self.audio_len = self.audio.duration_seconds
        print("SoundBoting, baby")


    # def robot(self):
    #     # calculate derivation in data for each wheel
    #     bot_move_left, bot_move_right = self.calc_deviation()
    #
    #     # robot.set_motors(bot_move_left, bot_move_right)
    #     # print('moving robot', bot_move_left, bot_move_right)
    #     self.sound(bot_move_left, bot_move_right)


    def play_sound(self, AI_data, duration):

        # 1. calc starting point from AI data
        starting_pos = self.calc_start_point(AI_data)

        # adds a bit of overlap with audio threading
        dur_ms = duration + 100
        end_pos_ms = starting_pos + dur_ms

        # concats slicing data
        audio_slice = self.audio[starting_pos: end_pos_ms]

        # plays audio
        play(audio_slice)

    def calc_start_point(self, incoming):
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        return (((incoming - 0) * ((self.audio_len * 1000) - 0)) / (1.0 - 0)) + 0


if __name__ == '__main__':
    snd = SoundBot()
    snd.play_sound(0.345, 0.75)
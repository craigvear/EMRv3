from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

# todo: pd sound player!!! as tab~


class SoundBot: # smooths the data as a thread class
    def __init__(self):
        # todo: if channel == 'left':

        # audio source variables
        audio_file = 'data/alfie.mp3'
        self.audio = AudioSegment.from_mp3(audio_file)
        # self.audio.pan(pan)
        audio_len = self.audio.duration_seconds
        self.audio_len_ms = audio_len * 1000
        print('SoundBoting, baby')
        print(f'audio length = {audio_len} seconds; = {self.audio_len_ms} ms')

    # def robot(self):
    #     # calculate derivation in data for each wheel
    #     bot_move_left, bot_move_right = self.calc_deviation()
    #
    #     # robot.set_motors(bot_move_left, bot_move_right)
    #     # print('moving robot', bot_move_left, bot_move_right)
    #     self.sound(bot_move_left, bot_move_right)

    def play_sound(self, ai_data, duration):

        # 1. calc starting point from AI data
        starting_pos = self.calc_start_point(ai_data)

        # adds a bit of overlap with audio threading
        dur_ms = duration + 100
        end_pos_ms = starting_pos + dur_ms

        # concats slicing data
        audio_slice = self.audio[starting_pos: end_pos_ms]

        print(f'SoundBot: audio params = {starting_pos}: {end_pos_ms}: {duration}')

        # plays audio
        play(audio_slice)

    def calc_start_point(self, incoming):
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        # new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
        return ( (incoming - 0) / (1 - 0) ) * (self.audio_len_ms - 0) + 0
        # return (((incoming - 0) * ((self.audio_len_ms) - 0)) / (1.0 - 0)) + 0

if __name__ == '__main__':
    snd = SoundBot()
    snd.play_sound(0.345, 0.75)
import time
import config
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

"""
takes the stored variables in config, and mixes then then smooths output
"""
class Robot(): # smooths the data as a thread class
    def __init__(self):
        # audio source variables
        audio_file = ('data/jarrett_snippet.wav')
        self.audio = AudioSegment.from_wav(audio_file)
        self.audio_len = self.audio.duration_seconds
        print("Roboting, baby")

        # temp moving vars
        self.old_left = 0
        self.old_left_sound = 0
        self.old_right = 0
        self.old_right_sound = 0

    def smooth(self, smoothing_dur, end_time):
        # slide between them at bang_timer ms per step
        while time.time() < end_time:
            # smoothing algo from Max/MSP slide object
            # y(n) = y(n - 1) + ((x(n) - y(n - 1)) / slide)

            current_l = config.left_wheel_move
            target_l = config.left_raw_data
            current_r = config.right_wheel_move
            target_r = config.right_raw_data

            duration = smoothing_dur
            self.interval = duration / 10

            # number of increments
            noi = duration / self.interval

            # split the delta w/ noi
            increment_value_l = (target_l - current_l) / noi
            increment_value_r = (target_r - current_r) / noi

            # smooth ouputs
            for _ in range(int(noi)):
                current_l += increment_value_l
                current_r += increment_value_r

                # wheel movement = adjsted value
                config.left_wheel_move = current_l
                config.right_wheel_move = current_r

                # make the robot move and sound
                self.robot()

    def robot(self):
        # calculate derivation in data for each wheel
        bot_move_left, bot_move_right = self.calc_deviation()

        # robot.set_motors(bot_move_left, bot_move_right)
        # print('moving robot', bot_move_left, bot_move_right)
        self.sound(bot_move_left, bot_move_right)

    def calc_deviation(self):
        # sets up temp vars for current params
        left = config.left_wheel_move
        right = config.right_wheel_move

        # subtracts new from old and difference = wheel move
        bot_move_left = left - self.old_left
        bot_move_right = right - self.old_right

        # make old vars the current move vars
        config.old_left = bot_move_left
        config.old_right = bot_move_right
        return bot_move_left, bot_move_right

    def sound(self, bot_move_left, bot_move_right):
        # round incoming numbers to 2 dp
        bot_move_left_round = round(bot_move_left, 1)
        bot_move_right_round = round(bot_move_right, 1)
        # print (bot_move_left, bot_move_right)
        poss_length = int(self.audio_len - (self.interval))

        # if changes or not to number
        if bot_move_left_round == self.old_left_sound:
            self.old_left_sound = bot_move_left_round
        else:
            # calc start position
            start_pos_ms = self.calc_start_point(bot_move_left, poss_length)

            # send params to play func
            self.play_sound(start_pos_ms)

            # makes old value = new
            self.old_left_sound = bot_move_left_round

        if bot_move_right_round == self.old_right_sound:
            self.old_right_sound = bot_move_right_round
        else:
            # calc start position
            start_pos_ms = self.calc_start_point(bot_move_right, poss_length)

            # send params to play func
            self.play_sound(start_pos_ms)

            # makes old value = new
            self.old_right_sound = bot_move_right_round

        # wait because its using simple_audio which only starts play
        time.sleep(self.interval)

    def play_sound(self, start_pos):
        # adds a bit of overlap with audio threading
        dur_ms = self.interval * 1000 + 100
        end_pos_ms = start_pos + dur_ms
        # print('play params = ', start_pos, dur_ms)

        # concats slicing data
        audio_slice = self.audio[start_pos: end_pos_ms]

        # plays audio
        play(audio_slice)

    def calc_start_point(self, incoming, poss_length):
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        return (((incoming - 0) * ((poss_length * 1000) - 0)) / (1.0 - 0)) + 0

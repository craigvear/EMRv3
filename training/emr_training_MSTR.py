import conv_training as CONV
import RNN_training as RNN
from glob import glob
from pandas import read_json

# consts
dataset_dir = glob('dataset/*.json')

# create array from JSON and pass to training
def create_array(location, feature, subfeature = None, subsubfeature = None):
    print(f'creating array for {feature}, location {location}')
    # reset temp array
    temp_array = []

    # for each JSON set in dataset parse feature and add to temp array
    for i_set in dataset_dir:
        df = read_json(i_set)

        # parse the values for each line of set
        for stuff in df.values:
            if subfeature == None:
                temp_array.append(stuff[location][feature])
            else:
                temp_array.append(stuff[location][feature][subfeature][subsubfeature])

    return temp_array


def main():
    # generate working arrays
    bitalino_array = create_array(4, 'bitalino')
    nose_array = create_array(4, 'skeleton', 'nose', 'x')

    # instantiate training class objects
    # conv = CONV.Training()
    rnn = RNN.Training()

    # go get 'em cowgirl
    print('starting to train')

    # # 1st RNN = affect in(y) - move out(x)
    # print('     1st RNN = affect in(y) - move out(x)')
    # conv.train(nose_array, bitalino_array, 'affect-move')
    #
    # # 2nd RNN = move in(y) - affect out(x)
    # print('     2nd RNN = move in(y) - affect out(x)')
    # conv.train(bitalino_array, nose_array, 'move-affect')

    # 3 skeleton (nose only)
    print('     3 skeleton (nose only)')
    rnn.train(nose_array, 'skeleton_data.nose.x')

    # 4 bitalino
    print('     4 bitalino')
    rnn.train(bitalino_array, 'bitalino')

if __name__ == "__main__":
    main()

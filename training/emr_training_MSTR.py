import conv_training as CONV
import RNN_training as RNN
from glob import glob
from pandas import read_json

# consts
dataset_dir = glob('dataset/*.json')


# create array from JSON and pass to

def create_array(location, feature):
    print(f'creating array for {feature}, location {location}')
    # reset temp array
    temp_array = []

    # for each JSON set in dataset parse feature and add to temp array
    for set in dataset_dir:
        df = read_json(set)

        # parse the values for each line of set
        for stuff in df.values:
            temp_array.append(stuff[location][feature])

    return temp_array


def main():
    # generate working arrays
    bitalino_array = create_array(4, 'bitalino')
    nose_array = create_array(4, 'skeleton.nose.x')



    # instantiate training class objects
    conv = CONV.Training()
    rnn = RNN.Training()

    # go get 'em cowgirl

    # 1st RNN = affect in(y) - move out(x)
    # x_train = conv.prep_sets(14) # skeleton.nose.x
    # y_train = conv.prep_sets(9) # bitalino
    conv.train(nose_array, bitalino_array, 'affect-move')

    # 2nd RNN = move in(y) - affect out(x)
    conv.train(bitalino_array, nose_array, 'move-affect')

    # 3 skeleton (nose only)
    rnn.prep_sets('skeleton_data.nose.x', 14)

    # 4 bitalino
    rnn.prep_sets('bitalino', 9)

if __name__ == "__main__":
    main()

import pretty_midi as pm
import librosa.display
import _pickle as pickle
from itertools import zip_longest
import matplotlib.pyplot as plt

midi = pm.PrettyMIDI('/home/just/Downloads/Piano-midi.de/train/alb_esp1.mid')
def plot_piano_roll(midi, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    # print(plt.isinteractive())
    # plt.interactive(True)
    pr = midi.get_piano_roll()
    librosa.display.specshow(midi.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))


with open('/home/just/Downloads/Piano-midi.de/Piano-midi.de.pickle','rb') as f:

    histo = pickle.load(f)
    # print(len(histo))
    # for i in histo:
    #     print(len(i))

    pr = histo['train'][0]
    print(len(pr))
    for i in pr:
        print(i)

    plot_piano_roll(midi,24,84)

    print('There are {} instruments'.format(len(midi.instruments)))
    print('First instrument is {}'.format(midi.instruments[0]))
    print('Second instrument is {}'.format(midi.instruments[1]))
    print('Instrument 1 has {} notes'.format(len(midi.instruments[0].notes)))
    print('Instrument 2 has {} notes'.format(len(midi.instruments[1].notes)))
    print(midi.get_beats(),len(midi.get_beats()))

    for i,j in zip_longest(midi.instruments[0].notes, midi.instruments[1].notes):
        print('left:{},right:{}'.format(i,j))

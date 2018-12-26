from settings import *
import numpy as np
import midi_functions as mf
import _pickle as pickle
import os
import sys
import pretty_midi as pm
import mido
from collections import Counter
    

def histo_of_all_songs():
    histo = [0]*128
    for path, subdirs, files in os.walk(tempo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
#            _name = _name[:-7]
            pianoroll = mf.get_pianoroll(_name, _path, fs)
            histo += np.sum(pianoroll, axis=1)
#            print(histo)
#            print(_name)
    return histo
        

def get_scales():
    # get all scales for every root note
    dia = tuple((0,2,4,5,7,9,11))
    diatonic_scales = []
    for i in range(0,12):
        diatonic_scales.append(tuple(np.sort((np.array(dia)+i)%12)))
    
    harm = tuple((0,2,4,5,8,9,11))
    harmonic_scales = []
    for i in range(0,12):
        harmonic_scales.append(tuple(np.sort((np.array(harm)+i)%12)))
    
    mel = tuple((0,2,4,6,8,9,11))
    melodic_scales = []
    for i in range(0,12):
        melodic_scales.append(tuple(np.sort((np.array(mel)+i)%12)))
    blue = tuple((0,3,5,6,7,10))
    blues_scales = []
    for i in range(0,12):
        blues_scales.append(tuple(np.sort((np.array(blue)+i)%12)))
    
    
    return diatonic_scales, harmonic_scales, melodic_scales, blues_scales


def get_shift(scale):
    diatonic_scales, harmonic_scales, melodic_scales, blues_scales = get_scales()
    if scale in diatonic_scales:
        return diatonic_scales.index(scale)
#    elif scale in harmonic_scales:
#        return harmonic_scales.index(scale)
#    elif scale in melodic_scales:
#        return melodic_scales.index(scale)
    else:
        return 'other'



def shift_midi_files(song_histo_folder,tempo_folder,shifted_folder):
    for path, subdirs, files in os.walk(song_histo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            tempo_path = tempo_folder+_path[len(song_histo_folder):]
            target_path = shifted_folder+_path[len(song_histo_folder):]
            song_histo = pickle.load(open(_path + _name, 'rb'))
            key = mf.histo_to_key(song_histo, key_n)
            shift = get_shift(key)
            _name = _name[:-7]
            if shift != 'other':
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                try:
                    mf.shift_midi(shift, _name, tempo_path, target_path)
                except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
                    exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
                    print(exception_str)


def count_scales():
    # get all scales for every root note
    diatonic_scales, harmonic_scales, melodic_scales, blues_scales = get_scales()

    scale_cntr = Counter()
    other_cntr = Counter()
    for path, subdirs, files in os.walk(song_histo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            song_histo = pickle.load(open(_path + _name, 'rb'))
            key = mf.histo_to_key(song_histo, key_n)
            if key in diatonic_scales:
                scale_cntr['diatonic'] +=1
                          
            elif key in harmonic_scales:
                scale_cntr['harmonic'] +=1
                          
            elif key in melodic_scales:
                scale_cntr['melodic'] +=1
            elif key[:-1] in blues_scales:
                scale_cntr['blues'] +=1
            else:
                scale_cntr['other'] += 1
                other_cntr[key] +=1
    return scale_cntr, other_cntr
    

def count_keys():
    key_cntr = Counter()
    for path, subdirs, files in os.walk(song_histo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            song_histo = pickle.load(open(_path + _name, 'rb'))
            key = mf.histo_to_key(song_histo, key_n)
            if key in key_cntr:
                key_cntr[key] +=1
            else:
                key_cntr[key] = 1                    
    return key_cntr


def save_song_histo_from_histo(histo_folder,song_histo_folder):
    for path, subdirs, files in os.walk(histo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = song_histo_folder+_path[len(histo_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            mf.load_histo_save_song_histo(_name, _path, target_path)


def save_index_from_chords(chords_folder,chords_index_folder):
    chord_to_index, index_to_chords = get_chord_dict()
    for path, subdirs, files in os.walk(chords_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = chords_index_folder+_path[len(chords_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            mf.chords_to_index_save(_name, _path, target_path, chord_to_index)
 

def get_chord_dict():
    chord_to_index = pickle.load(open(dict_path + chord_dict_name, 'rb'))
    index_to_chord = pickle.load(open(dict_path + chord_index_dict_name, 'rb'))
    return chord_to_index, index_to_chord

def save_idx_from_notes(roll_folder, note_index_folder):
    note_to_index,index_to_note = get_note_dict()
    for path, _, files in os.walk(roll_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            notes = pickle.load(open(_path + _name, 'rb'))
            note_index=[]
            for note in notes:
                if note in note_to_index :
                    note_index.append(note_to_index[note])
                else:
                    note_index.append(note_to_index[UNK])
            ## save index to target_path
            target_path = note_index_folder
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            pickle.dump(note_index,open(note_index_folder+_name, 'wb'))

def get_note_dict():
    note_to_index = pickle.load(open(dict_path + note_dict_name, 'rb'))
    index_to_note = pickle.load(open(dict_path + note_index_dict_name, 'rb'))
    return note_to_index, index_to_note



def make_chord_dict(chords_folder, num_chords):
    cntr = count_chords(chords_folder, num_chords)
    chord_to_index = dict()
    chord_to_index[UNK] = 0
    for chord, _ in cntr:
        chord_to_index[chord] = len(chord_to_index)
    index_to_chord = {v: k for k, v in chord_to_index.items()}
    pickle.dump(chord_to_index,open(dict_path + chord_dict_name , 'wb'))
    pickle.dump(index_to_chord, open(dict_path + chord_index_dict_name, 'wb'))
    return chord_to_index, index_to_chord

def count_chords(chords_folder, num_chords):
    chord_cntr = Counter()
    for path, subdirs, files in os.walk(chords_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            chords = pickle.load(open(_path + _name, 'rb'))
            for chord in chords:
                if chord in chord_cntr:
                    chord_cntr[chord] +=1
                else:
                    chord_cntr[chord] = 1                    
    return chord_cntr.most_common(n=num_chords-1)

def count_chords2(chords_folder, num_chords):
    chord_cntr = Counter()
    for path, subdirs, files in os.walk(chords_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _path = _path.replace('/shifted', '')
            _name = name.replace('\\', '/')
            chords = pickle.load(open(_path + _name, 'rb'))
            for chord in chords:
                if chord in chord_cntr:
                    chord_cntr[chord] +=1
                else:
                    chord_cntr[chord] = 1                    
    return chord_cntr

def make_note_dict(roll_folder,note_vocab_size):
    cntr = count_note(roll_folder,note_vocab_size)
    note_to_idx = dict()
    note_to_idx[UNK] = 0
    for note, _ in cntr:
        note_to_idx[note] = len(note_to_idx)
    idx_to_note = {v:k for k,v in note_to_idx.items()}
    pickle.dump(note_to_idx,open(dict_path + note_dict_name, 'wb'))
    pickle.dump(idx_to_note,open(dict_path + note_index_dict_name, 'wb'))
    return note_to_idx,idx_to_note


def count_note(roll_folder,note_vocab_size):
    cntr = Counter()
    for path,_,files in os.walk(roll_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            notes = pickle.load(open(_path + _name,'rb'))
            for note in notes:
                if note in cntr:
                    cntr[note] +=1
                else:
                    cntr[note] = 1
    print(len(cntr))
    return cntr.most_common(n = note_vocab_size - 1)


def save_chords_from_histo(histo_folder,chords_folder):
    for path, subdirs, files in os.walk(histo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = chords_folder+_path[len(histo_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            mf.load_histo_save_chords(chord_n, _name, _path, target_path)



def save_histo_oct_from_pianoroll_folder():
    #Not Used anymore!!
    for path, subdirs, files in os.walk(pickle_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = histo_folder+_path[len(pickle_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            mf.save_pianoroll_to_histo_oct(samples_per_bar,octave, _name, _path, target_path)


def save_histo_oct_from_midi_folder(tempo_folder,histo_folder):
    print(tempo_folder)
    for path, subdirs, files in os.walk(tempo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = histo_folder+_path[len(tempo_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            try:
                mf.midi_to_histo_oct(samples_per_bar,octave, fs, _name, _path, target_path)
            except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
                exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
                print(exception_str)
#                invalid_files_counter +=1


def pianoroll_folder():
    #Not Used anymore!!
    for path, subdirs, files in os.walk(tempo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = pickle_folder+_path[len(tempo_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            try:
                mf.save_pianoroll(_name, _path, target_path, fs)
            except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
                exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
                print(exception_str)
#                invalid_files_counter +=1


def note_ind_folder(tempo_folder,roll_folder):
    for path, subdirs, files in os.walk(tempo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = roll_folder+_path[len(tempo_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            try:
                mf.save_note_ind(_name, _path, target_path, fs)
            except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
                exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
                print(exception_str)
#                invalid_files_counter +=1

def change_tempo_folder(source_folder,tempo_folder):
    for path, subdirs, files in os.walk(source_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            target_path = tempo_folder+_path[len(source_folder):]
            if not os.path.exists(target_path):
                os.makedirs(target_path) 
            try:
                mf.change_tempo(_name, _path, target_path)
            except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
                exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
                print(exception_str)
#                invalid_files_counter +=1

def do_all_steps():
    

    print('changing Tempo')
    change_tempo_folder(source_folder,tempo_folder1) 
    
    print('histogramming')
    save_histo_oct_from_midi_folder(tempo_folder1,histo_folder1)

    print('make song histo')
    save_song_histo_from_histo(histo_folder1,song_histo_folder)
    
    print('shifting midi files')
    shift_midi_files(song_histo_folder,tempo_folder1,tempo_folder2)

    print('making note indexes')
    note_ind_folder(tempo_folder2,roll_folder)

    print('getting note dict')
    note_to_index,index_to_note = make_note_dict(roll_folder,note_vocab_size)

    print('converting notes to index seq')
    save_idx_from_notes(roll_folder,note_index_folder)

    print('histogramming')
    save_histo_oct_from_midi_folder(tempo_folder2,histo_folder2)

    print('extracting chords')
    save_chords_from_histo(histo_folder2,chords_folder)
    print('getting dictionary')
    chord_to_index, index_to_chord = make_chord_dict(chords_folder, num_chords)
    print('converting chords to index sequences')
    save_index_from_chords(chords_folder,chords_index_folder)


if __name__=="__main__":
    do_all_steps()
#    key_counter2 = count_keys()
#    scale_counter2, other_counter2 = count_scales()
#    shift_midi_files()
#    histo = histo_of_all_songs()
#    pickle.dump(histo, open('histo_all_songs.pickle', 'wb'))
#    chord_counter = count_chords(chords_folder, num_chords)
    print('done')
    
    

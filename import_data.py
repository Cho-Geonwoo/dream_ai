import os
import pandas as pd
import re
import math
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv

"""
Convert 30s mp3 files into mel-spectrograms.

A mel-spectrograms is a kind of time-frequency representation.
It is obtained from an audio signal by computing the Fourier transforms of short, overlapping windows.
Each of these Fourier transforms constitutes a frame.
These successive frames are then concatenated into a matrix to form the spectrogram.
"""
def create_spectrogram(verbose=0, mode=None):
    if mode == "Train":
        k = open("music_info.csv", "w",encoding='utf-8')
        wr = csv.writer(k)
        # if os.path.exists('Train_Spectogram_Images'):
        #     return
        # Get Genres and Track IDs from the tracks.csv file
        filename_metadata = "Dataset/fma_metadata/tracks.csv"
        tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
        tracks_array = tracks.values
        tracks_id_array = tracks_array[: , 0]
        tracks_genre_array = tracks_array[: , 40]
        tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
        tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

        folder_sample = "Dataset/fma_large"
        directories = [d for d in os.listdir(folder_sample)
                       if os.path.isdir(os.path.join(folder_sample, d))]
        counter = 0
        if(verbose > 0):
            print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Train_Spectogram_Images'):
            os.makedirs('Train_Spectogram_Images')
        for d in directories:
            label_directory = os.path.join(folder_sample, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if (f.endswith(".mp3") )]
            # Convert .mp3 files into mel-Spectograms
            for f in file_names:
                track_id = int(f[22:28])
                # print(tracks_id_array)
                track_index = list(tracks_id_array).index(track_id)
                if(str(tracks_genre_array[track_index, 0]) != '0'):
                    try:
                        y, sr = librosa.load(f)
                    #     melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
                    #     mel = librosa.power_to_db(melspectrogram_array)
                    # # Length and Width of Spectogram
                    #     fig_size = plt.rcParams["figure.figsize"]
                    #     fig_size[0] = float(mel.shape[1]) / float(100)
                    #     fig_size[1] = float(mel.shape[0]) / float(100)
                    #     plt.rcParams["figure.figsize"] = fig_size
                    #     plt.axis('off')
                    #     plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                    #     librosa.display.specshow(mel, cmap='gray_r')
                    #     plt.savefig("Train_Spectogram_Images/"+str(counter)+"_"+str(tracks_genre_array[track_index,0])+".jpg", bbox_inches=None, pad_inches=0)
                    #     plt.close()
                        wr.writerow([counter, f])
                        counter = counter + 1
                    except:
                        print("continue")
                        continue
        return

    elif mode == "Test":
        if os.path.exists('Test_Spectogram_Images'):
            return

        folder_sample = "Dataset/DLMusicTest_30"
        counter = 0
        if(verbose > 0):
            print("Converting mp3 audio files into mel Spectograms ...")
        # if not os.path.exists('Test_Sepctogram_Images'):
        #     os.makedirs('Test_Spectogram_Images')
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample)
                       if f.endswith(".wav")]
        print(file_names)
        # Convert .mp3 files into mel-Spectograms
        for f in file_names:
            test_id = re.search('(.+?).wav', f[23:]).group(1)
            print(test_id)
            y, sr = librosa.load(f)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)
            # Length and Width of Spectogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')
            plt.savefig("Test_Spectogram_Images/"+test_id+".jpg", cmap='gray_r', bbox_inches=None, pad_inches=0)
            plt.close()
        return

create_spectrogram(verbose=1, mode="Train")
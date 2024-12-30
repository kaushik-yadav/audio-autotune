import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import psola
import soundfile as sf
import scipy.signal as sig
import time


from functools import partial
from pathlib import Path


SEMITONES_IN_OCTAVE = 12

def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees

def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    return librosa.midi_to_hz(midi_note)

def closest_pitch_from_scale(f0, scale):
    """Return the pitch closest to f0 that belongs to the given scale"""
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % SEMITONES_IN_OCTAVE
    degree_id = np.argmin(np.abs(degrees - degree))
    degree_difference = degree - degrees[degree_id]
    midi_note -= degree_difference
    return librosa.midi_to_hz(midi_note)

def aclosest_pitch_from_scale(f0, scale):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch

def autotune(audio, sr, correction_function, plot=False):
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    corrected_f0 = correction_function(f0)

    if plot:
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def main(scale, index, file_name ="music/vocal.mp3"):
    # Hardcoded input values
    vocals_file = file_name  # Replace with your file path
    plot = True  # Set to True to generate a plot
    correction_method = "scale"  # Choose between 'closest' or 'scale'

    filepath = Path(vocals_file)

    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    if y.ndim > 1:
        y = y[0, :]

    correction_function = closest_pitch if correction_method == "closest" else \
        partial(aclosest_pitch_from_scale, scale=scale)

    pitch_corrected_y = autotune(y, sr, correction_function, plot)

    filepath = filepath.parent / (filepath.stem + f'_pitch_corrected_{index}' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)

if __name__ == '__main__':
    #scales = [
    #"C:maj", "D:maj", "E:maj", "F:maj", "G:maj", "A:maj", "B:maj",
    #"C:min", "D:min", "E:min", "F:min", "G:min", "A:min", "B:min",
    #"C:lyd", "D:lyd", "E:lyd", "F:lyd", "G:lyd", "A:lyd", "B:lyd",
    #"C:dor", "D:dor", "E:dor", "F:dor", "G:dor", "A:dor", "B:dor",
    #"C:loc", "D:loc", "E:loc", "F:loc", "G:loc", "A:loc", "B:loc"
    #]
    #video_url = "https://www.youtube.com/watch?v=8YeHPj9Qcw4"
    #path = "music"

    #download_audio(video_url, path)
    #count = 0
    #file_name = os.listdir(path)[0]

    scales = ["C:maj", "C:min", "F:lyd", "G:dor", "B:loc", "A:min"] 
    for i in range(len(scales)):
        main(scales[i], i + 1)#file_name

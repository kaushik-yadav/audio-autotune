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
from pydub import AudioSegment
from pytubefix import YouTube
from pytubefix.cli import on_progress



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

def autotune(audio, sr, correction_function, plot=False, scale = "1", output_folder = "output"):
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
        os.makedirs(output_folder, exist_ok=True)
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
        plt.savefig(os.path.join(output_folder, f'pitch_correction_{scale}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def download_audio(url, path):
        yt = YouTube(url, on_progress_callback=on_progress)
        ys = yt.streams.get_audio_only()
        audio_path = ys.download(output_path = path)
        return audio_path

def load_audio(filepath):
    try:
        # Use pydub to load the audio file
        audio = AudioSegment.from_file(filepath)
        sr = audio.frame_rate  # Sampling rate
        y = np.array(audio.get_array_of_samples())  # Convert audio to numpy array

        # If stereo, average channels to mono
        if audio.channels > 1:
            y = y.reshape((-1, audio.channels)).mean(axis=1)

        # Normalize the audio to the range [-1.0, 1.0] for librosa compatibility
        if np.issubdtype(y.dtype, np.integer):
            y = y / np.iinfo(y.dtype).max
        elif np.issubdtype(y.dtype, np.floating):
            y = y / np.abs(y).max()

        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {filepath}. Error: {e}")


def main(scale, index, audio_file_name, output_folder):
    """Main function to apply pitch correction."""
    plot = False
    filepath = Path(audio_file_name)
    correction_method = "scale"
    y, sr = load_audio(str(filepath))

    correction_function = closest_pitch if correction_method == "closest" else \
        partial(aclosest_pitch_from_scale, scale=scale)

    pitch_corrected_y = autotune(y, sr, correction_function, plot, scale, output_folder)

    output_file = Path(output_folder) / f'{filepath.stem}_pitch_corrected_{index}{filepath.suffix}'
    # Ensure the output file has a supported format (e.g., .wav)
    output_file = str(output_file.with_suffix(".wav"))
    sf.write(output_file, pitch_corrected_y, sr)
    

if __name__ == '__main__':
    #scales = [
    #"C:maj", "D:maj", "E:maj", "F:maj", "G:maj", "A:maj", "B:maj",
    #"C:min", "D:min", "E:min", "F:min", "G:min", "A:min", "B:min",
    #"C:lyd", "D:lyd", "E:lyd", "F:lyd", "G:lyd", "A:lyd", "B:lyd",
    #"C:dor", "D:dor", "E:dor", "F:dor", "G:dor", "A:dor", "B:dor",
    #"C:loc", "D:loc", "E:loc", "F:loc", "G:loc", "A:loc", "B:loc"
    #]
    video_url = "https://www.youtube.com/watch?v=8YeHPj9Qcw4"
    dir_name = "audio"
    os.makedirs(dir_name, exist_ok=True)

    audio_file_name = download_audio(video_url, dir_name)

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    scales = ["C:maj", "C:min", "F:lyd", "G:dor", "B:loc", "A:min"] 
    for i in range(len(scales)):
        main(scales[i], i + 1, audio_file_name, output_folder)
        break

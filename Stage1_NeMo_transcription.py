# Stage 1: ASR Inference (No LM) using a pretrained NeMo Conformer CTC model on Japanese data
from nemo.collections.asr.models import EncDecCTCModelBPE
import torchaudio
import glob
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
args = parser.parse_args()

# Define paths
AUDIO_DIRS = [args.input_dir]

# Gather all .wav files from all directories
audio_files = []
for d in AUDIO_DIRS:
    audio_files.extend(glob.glob(os.path.join(d, "*.wav")))
audio_files = sorted(audio_files)

# Load pretrained NeMo Conformer CTC model (Japanese-compatible)
asr_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_ja_conformer_transducer_large")
asr_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = asr_model.to(device)

# Helper function: load and preprocess audio (CPU)
def load_and_preprocess(file_path, target_sr):
    """
    Loads an audio file and resamples it to the target sample rate if needed.
    Returns the file path and the (possibly resampled) audio tensor.
    """
    audio, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        # Resample audio if sample rate does not match model's expected rate
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio = resampler(audio)
    return file_path, audio

# Parallel audio loading/preprocessing
def parallel_load(files, target_sr, max_workers=8):
    """
    Loads and preprocesses a list of audio files in parallel using multiple processes.
    Returns a list of (file_path, audio_tensor) tuples.
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_and_preprocess, f, target_sr) for f in files]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                # If loading fails, record the error with the file name
                results.append((os.path.basename(fut.arg[0]), f"[ERROR] {e}"))
    return results

# Main inference loop for a single audio file
def transcribe_audio(file_path):
    """
    Loads and preprocesses a single audio file, moves it to the correct device,
    and runs ASR transcription using the loaded model.
    Returns the transcribed text.
    """
    audio, _ = load_and_preprocess(file_path, asr_model.sample_rate)
    audio = audio.to(device)
    text = asr_model.transcribe([audio])[0]
    return text

def main():
    """
    For each audio directory, transcribes all .wav files and writes the results
    to a separate output file named after the directory.
    """
    for audio_dir in AUDIO_DIRS:
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        results = []
        for audio_path in audio_files:
            try:
                # Transcribe each audio file and collect results
                text = transcribe_audio(audio_path)
                results.append((os.path.basename(audio_path), text))
            except Exception as e:
                # Record errors for files that fail transcription
                results.append((os.path.basename(audio_path), f"[ERROR] {e}"))
        # Create output file name based on directory name
        dir_name = os.path.basename(os.path.normpath(audio_dir))
        output_file = f"/mnt/data/asr_raw_hypotheses_{dir_name}.txt"
        # Write results to output file
        with open(output_file, "w", encoding="utf-8") as f:
            for fname, hyp in results:
                f.write(f"{fname}\t{hyp}\n")

if __name__ == "__main__":
    main()

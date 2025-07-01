#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract discrete speech tokens for wav files using an ONNX model.

Example:
    python tools/wav2speech_token.py --onnx_path speech_tokenizer.onnx audio1.wav audio2.wav

This will create ``audio1.npy`` and ``audio2.npy`` next to the input wavs.
"""
import argparse
import logging
import os
import numpy as np
import onnxruntime
import torchaudio
import whisper


def load_and_resample(wav_path: str) -> torchaudio.Tensor:
    """Load audio file and resample to 16 kHz mono."""
    audio, sample_rate = torchaudio.load(wav_path, backend='soundfile')
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                               new_freq=16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def extract_token(wav_path: str, session: onnxruntime.InferenceSession) -> np.ndarray:
    """Return discrete speech token array for ``wav_path``."""
    audio = load_and_resample(wav_path)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s: %s', wav_path)
        return np.array([], dtype=np.int32)
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    token = session.run(None,
                        {session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                         session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten()
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='path to speech tokenizer ONNX model')
    parser.add_argument('wav_paths', nargs='+', help='wav files to process')
    args = parser.parse_args()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession(args.onnx_path,
                                           sess_options=option,
                                           providers=providers)

    for wav_path in args.wav_paths:
        token = extract_token(wav_path, session)
        npy_path = os.path.splitext(wav_path)[0] + '.npy'
        np.save(npy_path, token)
        logging.info('saved speech token to %s', npy_path)


if __name__ == '__main__':
    main()

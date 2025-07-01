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
"""Utility to convert speech tokens back to waveform.

This script is useful for debugging token generation. Given a token file and
an example prompt audio for speaker/style extraction, it reconstructs the
waveform using a pretrained CosyVoice model.
"""

import argparse
import os
import torch
import numpy as np
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.class_utils import get_model_type


def load_model(model_dir: str, fp16: bool):
    """Load model and frontend from ``model_dir``."""
    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir)

    # detect config file
    yaml_path = os.path.join(model_dir, "cosyvoice2.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(model_dir, "cosyvoice.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No config found in {model_dir}")

    with open(yaml_path, "r") as f:
        configs = load_hyperpyyaml(f)

    model_type = get_model_type(configs)
    tokenizer_model = (
        "speech_tokenizer_v2.onnx" if model_type is CosyVoice2Model else "speech_tokenizer_v1.onnx"
    )
    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        f"{model_dir}/campplus.onnx",
        f"{model_dir}/{tokenizer_model}",
        f"{model_dir}/spk2info.pt",
        configs.get("allowed_special", "all"),
    )

    if model_type is CosyVoice2Model:
        model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16)
    else:
        model = CosyVoiceModel(configs["llm"], configs["flow"], configs["hift"], fp16)
    model.load(f"{model_dir}/llm.pt", f"{model_dir}/flow.pt", f"{model_dir}/hift.pt")
    sample_rate = configs["sample_rate"]
    return model, frontend, sample_rate, model_type


def main(args):
    model, frontend, sample_rate, model_type = load_model(args.model_dir, args.fp16)

    # load token file
    if args.token_file.endswith(".npy"):
        token = np.load(args.token_file)
        token = torch.tensor(token, dtype=torch.int32)
    else:
        token = torch.load(args.token_file)
        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token, dtype=torch.int32)
    token = token.unsqueeze(0) if token.ndim == 1 else token

    # load prompt audio for embedding/feature extraction
    prompt_audio, sr = torchaudio.load(args.prompt_wav, backend="soundfile")
    if sr != 16000:
        prompt_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(prompt_audio)

    model_input = frontend.frontend_vc(prompt_audio, prompt_audio, sample_rate)
    prompt_token = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    embedding = model_input["flow_embedding"]

    uuid = "token_to_wav"
    if model_type is CosyVoice2Model:
        speech = model.token2wav(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            token_offset=0,
            uuid=uuid,
            finalize=True,
        )
    else:
        speech = model.token2wav(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            uuid=uuid,
            finalize=True,
        )

    torchaudio.save(args.output_wav, speech.cpu(), sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert speech tokens to waveform")
    parser.add_argument("--model_dir", type=str, required=True, help="pretrained model directory")
    parser.add_argument("--token_file", type=str, required=True, help="path to .npy or .pt token file")
    parser.add_argument("--prompt_wav", type=str, required=True, help="prompt audio used for embedding")
    parser.add_argument("--output_wav", type=str, required=True, help="output wav path")
    parser.add_argument("--fp16", action="store_true", help="load model with fp16")
    args = parser.parse_args()

    main(args)


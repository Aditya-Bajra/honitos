from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import json
import torch
from scipy.io.wavfile import write
from hifigan.env import AttrDict
from hifigan.models import Generator
from io import BytesIO

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def hifi_gan_inference(input_mel, checkpoint_file):
    print('Initializing Inference Process..')
    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Set MAX_WAV_VALUE if not present
    if 'MAX_WAV_VALUE' not in h:
        h.MAX_WAV_VALUE = 32768.0  # Adjust this value based on your requirements

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    # Load data from BytesIO
    buffer = BytesIO(input_mel)
    x = np.load(buffer)

    x = torch.FloatTensor(x).to(device)
    y_g_hat = generator(x)
    
    # Detach tensor before converting to numpy
    audio = y_g_hat.squeeze().detach().numpy()
    
    # Set MAX_WAV_VALUE if not present
    if 'MAX_WAV_VALUE' not in h:
        h.MAX_WAV_VALUE = 32768.0  # Adjust this value based on your requirements

    audio = audio * h.MAX_WAV_VALUE
    audio = audio.astype('int16')

    # Save audio to BytesIO
    output_buffer = BytesIO()
    write(output_buffer, h.sampling_rate, audio)
    
    return output_buffer.getvalue()
    
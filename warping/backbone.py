import torch

from music2latent.utils import download_model
from music2latent.hparams_inference import load_path_inference_default, \
                                           max_waveform_length_encode, \
                                           max_batch_size_encode, \
                                           max_batch_size_decode, \
                                           max_waveform_length_decode, \
                                           sigma_rescale
from music2latent.hparams import hop, freq_downsample_list
from music2latent.inference import encode_audio_inference, decode_latent_inference
from music2latent.models import UNet


class EncoderDecoder:
    def __init__(self, load_path_inference=None, device=None):
        download_model()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.load_path_inference = load_path_inference
        if load_path_inference is None:
            self.load_path_inference = load_path_inference_default
        self.get_models()
        
    def get_models(self):
        gen = UNet().to(self.device)
        checkpoint = torch.load(self.load_path_inference, map_location=self.device,
                                weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        self.gen = gen

    def encode(self, path_or_audio, max_waveform_length=None, max_batch_size=None, extract_features=False):
        '''
        path_or_audio: path of audio sample to encode or numpy array of waveform to encode
        max_waveform_length: maximum length of waveforms in the batch for encoding: tune it depending on the available GPU memory
        max_batch_size: maximum inference batch size for encoding: tune it depending on the available GPU memory

        WARNING! if input is numpy array of stereo waveform, it must have shape [waveform_samples, audio_channels]

        Returns latents with shape [audio_channels, dim, length]
        '''
        if max_waveform_length is None:
            max_waveform_length = max_waveform_length_encode
        if max_batch_size is None:
            max_batch_size = max_batch_size_encode
        return encode_audio_inference(path_or_audio, self, max_waveform_length, max_batch_size, device=self.device, extract_features=extract_features)
    
    def decode(self, latent, denoising_steps=1, max_waveform_length=None, max_batch_size=None):
        '''
        latent: numpy array of latents to decode with shape [audio_channels, dim, length]
        denoising_steps: number of denoising steps to use for decoding
        max_waveform_length: maximum length of waveforms in the batch for decoding: tune it depending on the available GPU memory
        max_batch_size: maximum inference batch size for decoding: tune it depending on the available GPU memory

        Returns numpy array of decoded waveform with shape [waveform_samples, audio_channels]
        '''
        if max_waveform_length is None:
            max_waveform_length = max_waveform_length_decode
        if max_batch_size is None: 
            max_batch_size = max_batch_size_decode
        return decode_latent_inference(latent, self, max_waveform_length, max_batch_size, diffusion_steps=denoising_steps, device=self.device)

    @torch.no_grad()
    def encode_features(self, features):
        downscaling_factor = 2**freq_downsample_list.count(0)
        max_feature_length = int(max_waveform_length_encode/hop) // downscaling_factor
        feature_length = features.shape[-1] 
        audio_channels = features.shape[0]

        pad_size = 0
        if feature_length > max_feature_length:
            # pad feature with copies of itself to be divisible by max_feature_length
            pad_size = max_feature_length - (feature_length % max_feature_length)
            # repeat features such that features.shape[-1] is higher than pad_size, then crop it such that lfeatures.shape[-1]=pad_size
            features_pad = torch.cat([features for _ in range(1+(pad_size//features.shape[-1]))], dim=-1)[:,:,:pad_size]
            features = torch.cat([features, features_pad], dim=-1)
            features = torch.split(features, max_feature_length, dim=-1)
            features = torch.cat(features, dim=0)

        latent = features
        for layer in self.gen.encoder.bottleneck_layers:
            latent = layer(latent)
                
        latent = self.gen.encoder.norm_out(latent)
        latent = self.gen.encoder.activation_out(latent)
        latent = self.gen.encoder.conv_out(latent)
        latent = self.gen.encoder.activation_bottleneck(latent)
        latent = latent / sigma_rescale

        if latent.shape[0] > 1:
            latent_ls = torch.split(latent, audio_channels, 0)
            latent = torch.cat(latent_ls, -1)
        latent = latent[:,:,:feature_length]

        return latent

from __future__ import print_function

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import gc
import config

label_csv = pd.read_csv(config.output_csv_with_audio_file_path)
label_csv = label_csv.sample(frac=1, random_state=1).reset_index(drop=True)

class AudioDataSet(Dataset):
    def __init__(self, range_start, range_stop):
        self.mixed_audio_names = []
        self.sep_audio_names = [[] for i in range(config.num_classes)]
        self.data_list = []
        self.target_list = [[] for i in range(range_start, range_stop)]

        set_count = 0

        if config.dataset_mode == 'all':
            for i in tqdm(range(range_start, range_stop)):
                self.mixed_audio_names.append(label_csv.iloc[i, -1])

                for j in range(config.num_classes):
                    self.sep_audio_names[j].append(label_csv.iloc[i, 1+2*j])
                
                data, target = load_data(self.mixed_audio_names, self.sep_audio_names, set_count)
                self.data_list.append(normalize(pre_process(data, config.processing_mode), None))

                for j in range(config.num_classes):
                    self.target_list[set_count].append(normalize(pre_process(target[j], config.processing_mode), self.data_list[set_count][1]))

                set_count += 1
        elif config.dataset_mode == 'indv':
            for i in tqdm(range(range_start, range_stop)):
                self.mixed_audio_names.append(label_csv.iloc[i, -1])

                for j in range(config.num_classes):
                    self.sep_audio_names[j].append(label_csv.iloc[i, 1+2*j])

    def __len__(self):
        return len(self.mixed_audio_names)

    def __getitem__(self, index):
        if config.dataset_mode == 'all':
            mixed_waveform = self.data_list[index]
            sep_waveform_list = self.target_list[index]
        elif config.dataset_mode == 'indv':
            data, target = load_data(self.mixed_audio_names, self.sep_audio_names, index)
            mixed_waveform = normalize(pre_process(data, config.processing_mode), None)
            sep_waveform_list = []
            
            for i in range(len(target)):
                sep_waveform_list.append(normalize(pre_process(target[i], config.processing_mode), mixed_waveform[1]))
        
        return (mixed_waveform, sep_waveform_list)

def load_data(mixed_audio_names, sep_audio_names, index):
    sep_waveform_list = []
    mixed_data, _ = librosa.load(config.project_dataset_input + mixed_audio_names[index], sr=config.sample_rate, mono=True)
    mixed_data = np.expand_dims(mixed_data, axis=0)
    mixed_waveform = np.zeros([1, config.waveform_duration * config.sample_rate])
   
    if mixed_data.shape[1] < mixed_waveform.shape[1]:
        mixed_waveform[0, :mixed_data.shape[1]] = mixed_data[0, :]
    else:
        mixed_waveform[:] = mixed_data[0, :mixed_waveform.shape[1]]

    del(mixed_data)
    gc.collect()
   
    for j in range(config.num_classes):
        sep_data, _ = librosa.load(config.project_dataset_output + sep_audio_names[j][index], sr=config.sample_rate, mono=True)
        sep_data = np.expand_dims(sep_data, axis=0)
        sep_waveform = np.zeros([1, config.waveform_duration * config.sample_rate])
        
        if sep_data.shape[1] < sep_waveform.shape[1]:
            sep_waveform[0, :sep_data.shape[1]] = sep_data[0, :]
        else:
            sep_waveform[:] = sep_data[0, :sep_waveform.shape[1]]
        
        sep_waveform_list.append(sep_waveform)

        del(sep_data)
        gc.collect()

    return (mixed_waveform, sep_waveform_list)

def pre_process(waveform, mode='stft'):
    if(mode == 'stft' or mode == 'stft_db'):
        waveform_sq = np.squeeze(waveform, axis=0)
        proc_waveform = librosa.stft(waveform_sq, config.stft_size, hop_length=config.stft_hop, window='hann')
        proc_waveform = np.expand_dims(proc_waveform, axis=2)
        proc_waveform = np.expand_dims(proc_waveform, axis=0)
        phase_stft_waveform = torch.from_numpy(np.angle(proc_waveform))
        
        if mode == 'stft_db':
            mag_stft_waveform = torch.from_numpy(librosa.amplitude_to_db(np.abs(proc_waveform)))
            proc_waveform = torch.cat((mag_stft_waveform, phase_stft_waveform), 3)
        elif mode == 'stft':
            mag_stft_waveform = torch.from_numpy(np.abs(proc_waveform))
            proc_waveform = torch.cat((mag_stft_waveform, phase_stft_waveform), 3)
    
    return proc_waveform

def post_process(proc_waveform_inp, mag_phase_params, mode='stft'):
    if(mode == 'stft' or mode == 'stft_db'):
        proc_waveform_denorm = denormalize(proc_waveform_inp, mag_phase_params)
        proc_waveform_denorm_np = proc_waveform_denorm.detach().numpy()
        proc_waveform_denorm_np = np.squeeze(proc_waveform_denorm_np, axis=0)
        stft_waveform_mag = proc_waveform_denorm_np[:, :, 0]
        stft_waveform_phase = proc_waveform_denorm_np[:, :, 1]

        if mode == 'stft_db':
            stft_waveform_mag = librosa.db_to_amplitude(stft_waveform_mag)

        proc_waveform = stft_waveform_mag * np.exp(1j*stft_waveform_phase)
        waveform = librosa.istft(proc_waveform, hop_length=config.stft_hop, window='hann')
        waveform = torch.from_numpy(np.expand_dims(waveform, axis=0))

    return waveform

def normalize(proc_waveform, mag_phase_params=None, mode='min-max'):
    output_list = []

    if mode == 'min-max':
        if mag_phase_params is None:
            min_mag = torch.min(proc_waveform[:, :, :, 0:1])
            max_mag = torch.max(proc_waveform[:, :, :, 0:1])
            min_phase = torch.min(proc_waveform[:, :, :, 1:2])
            max_phase = torch.max(proc_waveform[:, :, :, 1:2])
        else:
            min_mag = mag_phase_params[0]
            max_mag = mag_phase_params[1]
            min_phase = mag_phase_params[2]
            max_phase = mag_phase_params[3]

        if torch.allclose(max_mag, min_mag, equal_nan=True):
            proc_waveform_mag_norm = (proc_waveform[:, :, :, 0:1] - min_mag)
        else:
            proc_waveform_mag_norm = (proc_waveform[:, :, :, 0:1] - min_mag) / (max_mag - min_mag)

        if torch.allclose(max_phase, min_phase, equal_nan=True):
            proc_waveform_phase_norm = (proc_waveform[:, :, :, 1:2] - min_phase)
        else:
            proc_waveform_phase_norm = (proc_waveform[:, :, :, 1:2] - min_phase) / (max_phase - min_phase)

        proc_waveform = torch.cat((proc_waveform_mag_norm, proc_waveform_phase_norm), 3)
        output_list.append(proc_waveform)
        mag_phase_params = torch.tensor([min_mag, max_mag, min_phase, max_phase])
        output_list.append(mag_phase_params)
    elif mode == 'cmvn':
        if mag_phase_params is None:
            mean_mag = torch.mean(proc_waveform[:, :, :, 0:1])
            stdev_mag = torch.std(proc_waveform[:, :, :, 0:1])
            mean_phase = torch.mean(proc_waveform[:, :, :, 1:2])
            stdev_phase = torch.std(proc_waveform[:, :, :, 1:2])
        else:
            mean_mag = mag_phase_params[0]
            stdev_mag = mag_phase_params[1]
            mean_phase = mag_phase_params[2]
            stdev_phase = mag_phase_params[3]

        if np.isnan(stdev_mag.item()) or stdev_mag.item() == 0:
            proc_waveform_mag_norm = (proc_waveform[:, :, :, 0:1] - mean_mag)
        else:
            proc_waveform_mag_norm = (proc_waveform[:, :, :, 0:1] - mean_mag) / (stdev_mag)

        if np.isnan(stdev_phase.item()) or stdev_phase.item() == 0:
            proc_waveform_phase_norm = (proc_waveform[:, :, :, 1:2] - mean_phase)
        else:
            proc_waveform_phase_norm = (proc_waveform[:, :, :, 1:2] - mean_phase) / (stdev_phase)

        proc_waveform = torch.cat((proc_waveform_mag_norm, proc_waveform_phase_norm), 3)
        output_list.append(proc_waveform)
        mag_phase_params = torch.tensor([mean_mag, stdev_mag, mean_phase, stdev_phase])
        output_list.append(mag_phase_params)
    
    return output_list

def denormalize(proc_waveform_inp, mag_phase_params, mode='min-max'):
    if mode == 'min-max':
        if torch.allclose(mag_phase_params[1], mag_phase_params[0], equal_nan=True):
            proc_waveform_mag_denorm = proc_waveform_inp[:, :, :, 0:1] + mag_phase_params[0]
        else:
            proc_waveform_mag_denorm = (proc_waveform_inp[:, :, :, 0:1] * (mag_phase_params[1] - mag_phase_params[0])) + mag_phase_params[0]

        if torch.allclose(mag_phase_params[3], mag_phase_params[2], equal_nan=True):
            proc_waveform_phase_denorm = proc_waveform_inp[:, :, :, 1:2] + mag_phase_params[2]
        else:
            proc_waveform_phase_denorm = (proc_waveform_inp[:, :, :, 1:2] * (mag_phase_params[3] - mag_phase_params[2])) + mag_phase_params[2]

        proc_waveform_denorm = torch.cat((proc_waveform_mag_denorm, proc_waveform_phase_denorm), 3)
    elif mode == 'cmvn':
        if torch.allclose(mag_phase_params[1], mag_phase_params[0], equal_nan=True):
            proc_waveform_mag_denorm = proc_waveform_inp[:, :, :, 0:1] + mag_phase_params[0]
        else:
            proc_waveform_mag_denorm = (proc_waveform_inp[:, :, :, 0:1] * (mag_phase_params[1])) + mag_phase_params[0]

        if torch.allclose(mag_phase_params[3], mag_phase_params[2], equal_nan=True):
            proc_waveform_phase_denorm = proc_waveform_inp[:, :, :, 1:2] + mag_phase_params[2]
        else:
            proc_waveform_phase_denorm = (proc_waveform_inp[:, :, :, 1:2] * (mag_phase_params[3])) + mag_phase_params[2]

        proc_waveform_denorm = torch.cat((proc_waveform_mag_denorm, proc_waveform_phase_denorm), 3)

    return proc_waveform_denorm

def build_batch_dataset():
    dataset_size = len(pd.read_csv(config.output_csv_with_audio_file_path))

    print("Preparing Autoenconder Train Dataset...")
    train_set_size = int(config.train_set_percent * dataset_size)
    per_batch_size = int(train_set_size / config.train_batch_size)
    final_train_batch_size = 0
    
    for i in range(config.train_batch_size):
        train_set = AudioDataSet(i * per_batch_size, (i + 1) * per_batch_size)
        np.save(config.project_folder+"train_set_b"+str(i)+".npy", train_set)
        del(train_set)
        gc.collect()
    if per_batch_size * config.train_batch_size != train_set_size:
        train_set = AudioDataSet(config.train_batch_size * per_batch_size, train_set_size)
        np.save(config.project_folder+"train_set_b"+str(config.train_batch_size)+".npy", train_set)
        del(train_set)
        gc.collect()

    print("Preparing Autoenconder Dev Dataset...")
    dev_set_size = int(config.dev_set_percent * dataset_size)
    per_batch_size = int(dev_set_size / config.dev_batch_size)
    final_dev_batch_size = 0
    
    for i in range(config.dev_batch_size):
        dev_set = AudioDataSet(i * per_batch_size + train_set_size, (i + 1) * per_batch_size + train_set_size)
        np.save(config.project_folder+"dev_set_b"+str(i)+".npy", dev_set)
        del(dev_set)
        gc.collect()
    if (per_batch_size * config.dev_batch_size != dev_set_size) or (dev_set_size + train_set_size != dataset_size):
        dev_set = AudioDataSet(config.dev_batch_size * per_batch_size + train_set_size, dataset_size)
        np.save(config.project_folder+"dev_set_b"+str(config.dev_batch_size)+".npy", dev_set)
        del(dev_set)
        gc.collect()

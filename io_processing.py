from __future__ import print_function

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
hann_window = torch.hann_window(config.hann_window_length)

class AudioDataSet(Dataset):
	def __init__(self, range_start, range_stop):
		label_csv = pd.read_csv(config.output_csv_with_audio_file_path)
		self.mixed_audio_names = []
		self.sep_audio_names = [[] for i in range(config.num_classes)]
		for i in range(range_start, range_stop):
			if label_csv.iloc[i, 0] != "Crowd" and label_csv.iloc[i, 4] != "Car horn":
				self.mixed_audio_names.append(label_csv.iloc[i, 1][:-4]+"_"+label_csv.iloc[i, 3][:-4]+"_"+label_csv.iloc[i, 5])
				for j in range(config.num_classes):
					self.sep_audio_names[j].append(label_csv.iloc[i, 1+2*j])

	def __getitem__(self, index):
		sample_rate = 480000
		resample_rate = int(sample_rate/5)
		sep_waveform_list = []
		mixed_data = torchaudio.load(config.project_dataset_input + self.mixed_audio_names[index])
		mixed_waveform = mixed_data[0].reshape(1, -1)
		mixed_waveform_pad = torch.zeros([1, sample_rate])
		if mixed_waveform.numel() < sample_rate:
			mixed_waveform_pad[0,:mixed_waveform.numel()] = mixed_waveform[0,:] 
		else:
			mixed_waveform_pad[:] = mixed_waveform[:]
		#mixed_waveform_pad = torchaudio.transforms.Resample(sample_rate, resample_rate)(mixed_waveform_pad)
		mixed_waveform_pad_formatted = torch.zeros([1, resample_rate])
		mixed_waveform_pad_formatted[0,:resample_rate] = mixed_waveform_pad[0,:resample_rate] #take every 2nd sample of soundData
		mixed_waveform = PreProcessData(mixed_waveform_pad_formatted)
		for j in range(config.num_classes):
			sep_data = torchaudio.load(config.project_dataset_output + self.sep_audio_names[j][index])
			sep_waveform = sep_data[0].reshape(1, -1)
			sep_waveform_pad = torch.zeros([1, sample_rate])
			if sep_waveform.numel() < sample_rate:
				sep_waveform_pad[0,:sep_waveform.numel()] = sep_waveform[0,:] 
			else:
				sep_waveform_pad[:] = sep_waveform[:]
			#sep_waveform_pad = torchaudio.transforms.Resample(sample_rate, resample_rate)(sep_waveform_pad)
			sep_waveform_pad_formatted = torch.zeros([1, resample_rate])
			sep_waveform_pad_formatted[0,:resample_rate] = sep_waveform_pad[0,:resample_rate] #take every 2nd sample of soundData
			sep_waveform_list.append(PreProcessData(sep_waveform_pad_formatted))

		return (mixed_waveform, sep_waveform_list)

	def __len__(self):
		return len(self.mixed_audio_names)

def PreProcessData(waveform):
	stft_waveform = torch.stft(waveform, config.stft_size, window = hann_window)
	#log_stft_waveform = torch.log(stft_waveform)

	return stft_waveform

def PostProcessData(stft_waveform_inp):
	#stft_waveform = torch.exp(log_stft_waveform_inp)
	waveform = torchaudio.functional.istft(stft_waveform_inp, config.stft_size, window = hann_window)

	return waveform

dataset_size = len(pd.read_csv(config.output_csv_with_audio_file_path))

train_set = AudioDataSet(0, int(config.train_set_percent * dataset_size))
dev_set = AudioDataSet(int(config.train_set_percent * dataset_size), dataset_size)

#train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.batch_size, shuffle = True, **dataloader_kwargs)
#dev_loader = torch.utils.data.DataLoader(dev_set, batch_size = config.batch_size, shuffle = True, **dataloader_kwargs)
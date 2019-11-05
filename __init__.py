from __future__ import print_function

import os
import sys
import mir_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import config
import io_processing

class AutoEncoderModel(nn.Module):
	def __init__(self):
		super(AutoEncoderModel, self).__init__()

		self.encode_cnn = nn.Sequential(
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_inp_channels, config.num_conv_horiz_channels, config.conv_horiz_filter_size),
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_conv_vert_channels, config.conv_vert_filter_size)
		)
		
		self.shared_fcl = nn.Sequential(
			nn.Linear(config.num_fcc_in_features, config.num_fcc_out_features),
			nn.ReLU()
		)

		self.fcl_speech = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_speech = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_baby_cry = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_baby_cry = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_siren = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_siren = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.fcl_dog = nn.Sequential(
			nn.Linear(config.num_fcc_out_features, config.num_fcc_in_features),
			nn.ReLU()
		)

		self.deconv_dog = nn.Sequential(
			nn.ZeroPad2d((0, 0, config.padding_vert_size[0], config.padding_vert_size[1])),
			nn.Conv2d(config.num_conv_vert_channels, config.num_conv_horiz_channels, config.conv_vert_filter_size),
			nn.ZeroPad2d((config.padding_horiz_size[0], config.padding_horiz_size[1], 0, 0)),
			nn.Conv2d(config.num_conv_horiz_channels, config.num_inp_channels, config.conv_horiz_filter_size)
		)

		self.out_ReLU = nn.ReLU();

	def forward(self, inp):
		encode_out = self.encode_cnn(inp)
		shared_fc_out = self.shared_fcl(encode_out)

		decode_speech = self.fcl_speech(shared_fc_out)
		decode_speech = self.deconv_speech(decode_speech)

		decode_baby_cry = self.fcl_speech(shared_fc_out)
		decode_baby_cry = self.deconv_baby_cry(decode_baby_cry)

		decode_dog = self.fcl_speech(shared_fc_out)
		decode_dog = self.deconv_dog(decode_dog)

		decode_siren = self.fcl_speech(shared_fc_out)
		decode_siren = self.deconv_siren(decode_siren)

		concat_out = torch.cat((decode_speech, decode_baby_cry, decode_siren, decode_dog), 1)
		concat_out = self.out_ReLU(concat_out)

		return concat_out

def train(model):
	model.train()
	
	loss_list = []

	for epoch in range(config.num_epochs):
		train_set_count = 0
		for (data, target) in (io_processing.train_set):
			data = torch.FloatTensor(data)
			model_out = model(data.permute(0, 3, 1, 2))
			
			for i in range(len(target)):
				target[i] = torch.FloatTensor(target[i])
				target[i] = target[i][:, :, :].permute(0, 3, 1, 2)
			
			concat_target = torch.cat((target[0], target[1], target[2], target[3]), 1)
			
			loss = criterion(model_out, concat_target)
			loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if train_set_count % 1 == 0:
				print('Train Epoch: {} \tLoss: {:.6f}'.format(
					epoch, loss
				))

			if train_set_count % 4 == 0:
				break

			train_set_count += 1
			
	torch.save(model.state_dict(), config.log_final_state_folder + "auto_encoder_model_state.pt")
	torch.save(model.state_dict(), config.log_complete_model_folder + "auto_encoder_model.pt")

def eval(model):
	model.eval()

	sdr_list = []
	isr_list = []
	sir_list = []
	sar_list = []

	for (data, target) in (io_processing.dev_set):
		data = torch.FloatTensor(data)
		model_out = model(data.permute(0, 3, 1, 2))
		model_out = model_out.permute(0, 2, 3, 1)

		model_estimate = []

		for i in range(len(target)):
			model_estimate[i] = io_processing.PostProcessData(model_out[:,:,:,i*(2):i*(2)+2])	

			target[i] = torch.FloatTensor(target[i])
			target[i] = io_processing.PostProcessData(target[i])
		
		concat_estimate = torch.cat((model_estimate[0], model_estimate[1], model_estimate[2], model_estimate[3]), 1)
		concat_target = torch.cat((target[0], target[1], target[2], target[3]), 1)

		[sdr, isr, sir, sar, _] = mir_eval.separation.bss_eval_images(concat_target.detach().numpy(), concat_estimate.detach().numpy())
		
		sdr_list.append(sdr)
		isr_list.append(isr)
		sir_list.append(sir)
		sar_list.append(sar)

	return (sdr_list, isr_list, sir_list, sar_list)

def init_log():
	if not os.path.isdir(config.log_folder):
		os.mkdir(config.log_folder)
	if not os.path.isdir(config.log_final_state_folder):
		os.mkdir(config.log_final_state_folder)
	if not os.path.isdir(config.log_indv_state_folder):
		os.mkdir(config.log_indv_state_folder)
	if not os.path.isdir(config.log_complete_model_folder):
		os.mkdir(config.log_complete_model_folder)

if __name__ == '__main__':
	init_log()

	model = AutoEncoderModel()
	optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
	criterion = nn.MSELoss()

	if sys.argv[1] == '--train' or sys.argv[1] == '--t':
		print("Training the Autoencoder Model...")
		train(model)
	elif sys.argv[1] == '--eval' or sys.argv[1] == '--e':
		print("Evaluating the Autoenconder Model...")
		eval(model)
    else:
        print("%s --train to train the autoencoder model"%sys.argv[0])
        print("%s --eval to evaluate the autoencoder model"%sys.argv[0])
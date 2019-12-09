from __future__ import print_function

import os
import sys
import mir_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
import soundfile as sf
import glob
import math
import random
import argparse
import gc
import config
import io_processing

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-20

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()

        self.encode_cnn = nn.Sequential(
            nn.Conv2d(
                config.num_inp_channels,
                config.num_conv_horiz_channels,
                config.conv_horiz_filter_size
            ),
            nn.Conv2d(
                config.num_conv_horiz_channels,
                config.num_conv_vert_channels,
                config.conv_vert_filter_size
            )
        )

        self.shared_fcl = nn.Sequential(
            nn.Linear(
                config.num_fcc_in_features,
                config.num_fcc_out_features
            ),
            nn.ReLU()
        )

        self.fcl_baby_cry = nn.Sequential(
            nn.Linear(
                config.num_fcc_out_features,
                config.num_fcc_in_features
            ),
            nn.ReLU()
        )

        self.deconv_baby_cry = nn.Sequential(
            nn.ConvTranspose2d(
                config.num_conv_vert_channels,
                config.num_conv_horiz_channels,
                config.conv_vert_filter_size
            ),
            nn.ConvTranspose2d(
                config.num_conv_horiz_channels,
                config.num_inp_channels,
                config.conv_horiz_filter_size
            )
        )

        self.fcl_siren = nn.Sequential(
            nn.Linear(
                config.num_fcc_out_features,
                config.num_fcc_in_features
            ),
            nn.ReLU()
        )

        self.deconv_siren = nn.Sequential(
            nn.ConvTranspose2d(
                config.num_conv_vert_channels,
                config.num_conv_horiz_channels,
                config.conv_vert_filter_size
            ),
            nn.ConvTranspose2d(
                config.num_conv_horiz_channels,
                config.num_inp_channels,
                config.conv_horiz_filter_size
            )
        )

        self.fcl_dog = nn.Sequential(
            nn.Linear(
                config.num_fcc_out_features,
                config.num_fcc_in_features
            ),
            nn.ReLU()
        )

        self.deconv_dog = nn.Sequential(
            nn.ConvTranspose2d(
                config.num_conv_vert_channels,
                config.num_conv_horiz_channels,
                config.conv_vert_filter_size
            ),
            nn.ConvTranspose2d(
                config.num_conv_horiz_channels,
                config.num_inp_channels,
                config.conv_horiz_filter_size
            )
        )

        self.fcl_speech = nn.Sequential(
            nn.Linear(
                config.num_fcc_out_features,
                config.num_fcc_in_features
            ),
            nn.ReLU()
        )

        self.deconv_speech = nn.Sequential(
            nn.ConvTranspose2d(
                config.num_conv_vert_channels,
                config.num_conv_horiz_channels,
                config.conv_vert_filter_size
            ),
            nn.ConvTranspose2d(
                config.num_conv_horiz_channels,
                config.num_inp_channels,
                config.conv_horiz_filter_size
            )
        )

        self.fcl_others = nn.Sequential(
            nn.Linear(
                config.num_fcc_out_features,
                config.num_fcc_in_features
            ),
            nn.ReLU()
        )

        self.deconv_others = nn.Sequential(
            nn.ConvTranspose2d(
                config.num_conv_vert_channels,
                config.num_conv_horiz_channels,
                config.conv_vert_filter_size
            ),
            nn.ConvTranspose2d(
                config.num_conv_horiz_channels,
                config.num_inp_channels,
                config.conv_horiz_filter_size
            )
        )

        self.out_ReLU = nn.ReLU()

    def forward(self, inp):
        encode_out = self.encode_cnn(inp)
        encode_out = encode_out.view(config.batch_size, -1)
        shared_fc_out = self.shared_fcl(encode_out)

        decode_baby_cry = self.fcl_baby_cry(shared_fc_out)
        decode_baby_cry = decode_baby_cry.view(
            -1, config.num_conv_vert_channels, config.num_enc_out_features, 1)
        decode_baby_cry = self.deconv_baby_cry(decode_baby_cry)

        decode_dog = self.fcl_dog(shared_fc_out)
        decode_dog = decode_dog.view(
            -1, config.num_conv_vert_channels, config.num_enc_out_features, 1)
        decode_dog = self.deconv_dog(decode_dog)

        decode_siren = self.fcl_siren(shared_fc_out)
        decode_siren = decode_siren.view(
            -1, config.num_conv_vert_channels, config.num_enc_out_features, 1)
        decode_siren = self.deconv_siren(decode_siren)

        decode_speech = self.fcl_speech(shared_fc_out)
        decode_speech = decode_speech.view(
            -1, config.num_conv_vert_channels, config.num_enc_out_features, 1)
        decode_speech = self.deconv_speech(decode_speech)

        decode_others = self.fcl_others(shared_fc_out)
        decode_others = decode_others.view(
            -1, config.num_conv_vert_channels, config.num_enc_out_features, 1)
        decode_others = self.deconv_others(decode_others)

        concat_out = torch.cat((decode_baby_cry, decode_dog, decode_siren, decode_speech, decode_others), 1)
        concat_out = self.out_ReLU(concat_out)

        return concat_out

def compute_loss(data, target, criterion, model_out):
    model_baby_cry = model_out[:, :1, :, :]
    model_dog = model_out[:, 1:2, :, :]
    model_siren = model_out[:, 2:3, :, :]
    model_speech = model_out[:, 3:4, :, :]
    model_others = model_out[:, 4:5, :, :]

    sum_model = model_baby_cry + model_dog + model_siren + model_speech + model_others

    mask_baby_cry = model_baby_cry / sum_model
    mask_dog = model_dog / sum_model
    mask_siren = model_siren / sum_model
    mask_speech = model_speech / sum_model
    mask_others = model_others / sum_model

    calc_baby_cry = data * mask_baby_cry
    calc_dog = data * mask_dog
    calc_siren = data * mask_siren
    calc_speech = data * mask_speech
    calc_others = data * mask_others

    loss_other_baby_cry = 0
    loss_other_dog = 0
    loss_other_siren = 0
    loss_other_speech = 0
    loss_other_others = 0

    criterion_loss_baby_cry = criterion(calc_baby_cry, target[:, :1, :, :])
    loss_other_baby_cry += config.alpha_baby_cry * criterion(calc_baby_cry, target[:, 1:2, :, :])
    loss_other_baby_cry += config.alpha_baby_cry * criterion(calc_baby_cry, target[:, 2:3, :, :])
    loss_other_baby_cry += config.alpha_baby_cry * criterion(calc_baby_cry, target[:, 3:4, :, :])
    loss_other_baby_cry += config.alpha_baby_cry * criterion(calc_baby_cry, target[:, 4:5, :, :])

    criterion_loss_dog = criterion(calc_dog, target[:, 1:2, :, :])
    loss_other_dog += config.alpha_dog * criterion(calc_dog, target[:, :1, :, :])
    loss_other_dog += config.alpha_dog * criterion(calc_dog, target[:, 2:3, :, :])
    loss_other_dog += config.alpha_dog * criterion(calc_dog, target[:, 3:4, :, :])
    loss_other_dog += config.alpha_dog * criterion(calc_dog, target[:, 4:5, :, :])

    criterion_loss_siren = criterion(calc_siren, target[:, 2:3, :, :])
    loss_other_siren += config.alpha_siren * criterion(calc_siren, target[:, :1, :, :])
    loss_other_siren += config.alpha_siren * criterion(calc_siren, target[:, 1:2, :, :])
    loss_other_siren += config.alpha_siren * criterion(calc_siren, target[:, 3:4, :, :])
    loss_other_siren += config.alpha_siren * criterion(calc_siren, target[:, 4:5, :, :])

    criterion_loss_speech = criterion(calc_speech, target[:, 3:4, :, :])
    loss_other_speech += config.alpha_speech * criterion(calc_speech, target[:, :1, :, :])
    loss_other_speech += config.alpha_speech * criterion(calc_speech, target[:, 1:2, :, :])
    loss_other_speech += config.alpha_speech * criterion(calc_speech, target[:, 2:3, :, :])
    loss_other_speech += config.alpha_speech * criterion(calc_speech, target[:, 4:5, :, :])

    criterion_loss_others = criterion(calc_others, target[:, 4:5, :, :])
    loss_other_others += config.alpha_others * criterion(calc_others, target[:, :1, :, :])
    loss_other_others += config.alpha_others * criterion(calc_others, target[:, 1:2, :, :])
    loss_other_others += config.alpha_others * criterion(calc_others, target[:, 2:3, :, :])
    loss_other_others += config.alpha_others * criterion(calc_others, target[:, 3:4, :, :])

    return (criterion_loss_baby_cry, criterion_loss_dog, criterion_loss_siren, criterion_loss_speech,
           criterion_loss_others, loss_other_baby_cry, loss_other_dog, loss_other_siren, loss_other_speech,
           loss_other_others)

def train(model, criterion, optimizer, train_set):
    model.train()

    train_batch_size = 1

    if config.dataset_mode == 'all':
        train_batch_size = len(glob.glob1(config.project_folder,"train_set_b*"))

    loss_list = []
    loss_baby_cry_list = []
    loss_dog_list = []
    loss_siren_list = []
    loss_speech_list = []
    loss_others_list = []

    random.seed(0)

    for epoch in range(config.num_epochs):
        train_set_count = 0
        loss = 0

        avg_loss = 0
        avg_loss_baby_cry = 0
        avg_loss_dog = 0
        avg_loss_siren = 0
        avg_loss_speech = 0
        avg_loss_others = 0

        optimizer.zero_grad()

        shuffle_batch = list(range(train_batch_size))
        random.shuffle(shuffle_batch)

        for batch in shuffle_batch:
            if config.dataset_mode == 'all':
                train_set = np.load(config.project_folder+"train_set_b"+str(batch)+".npy", allow_pickle=True)

            for (norm_data, norm_target) in train_set:
                train_data = norm_data[0][:, :, :, :1].permute(0, 3, 2, 1)

                train_data = train_data.to(DEVICE)

                model_out = model(train_data) + EPS

                train_target = []

                for i, element in enumerate(norm_target):
                    train_target.append(norm_target[i][0][:, :, :, :1].permute(0, 3, 2, 1))

                concat_target = torch.cat((train_target[0], train_target[1], train_target[2], train_target[3], train_target[4]), 1)
                concat_target = concat_target.to(DEVICE)

                (loss_baby_cry, loss_dog, loss_siren, loss_speech, loss_others, loss_other_baby_cry, loss_other_dog, loss_other_siren, loss_other_speech, loss_other_others) = compute_loss(train_data, concat_target, criterion, model_out)
                loss = abs(loss_baby_cry + loss_dog + loss_siren + loss_speech + loss_others - loss_other_baby_cry - loss_other_dog - loss_other_siren - loss_other_speech - loss_other_others)

                avg_loss += loss
                avg_loss_baby_cry += loss_baby_cry
                avg_loss_dog += loss_dog
                avg_loss_siren += loss_siren
                avg_loss_speech += loss_speech
                avg_loss_others += loss_others

                if torch.isnan(loss):
                    optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_set_count += 1

            if config.dataset_mode == 'all':
                del(train_set)
                gc.collect()

        avg_loss /= train_set_count
        avg_loss_baby_cry /= train_set_count
        avg_loss_dog /= train_set_count
        avg_loss_siren /= train_set_count
        avg_loss_speech /= train_set_count
        avg_loss_others /= train_set_count

        if (epoch % config.epoch_log == 0):
            print('Train Epoch: {} \tAvg. Loss: {:.2f} \tAvg. Baby Loss: {:.2f} \tAvg. Dog Loss: {:.2f} \tAvg. Siren Loss: {:.2f} \tAvg. Speech Loss: {:.2f} \tAvg. Others Loss: {:.2f}'.format(
                epoch, avg_loss, avg_loss_baby_cry, avg_loss_dog, avg_loss_siren, avg_loss_speech, avg_loss_others
            ))

            loss_list.append(avg_loss)
            loss_baby_cry_list.append(avg_loss_baby_cry)
            loss_dog_list.append(avg_loss_dog)
            loss_siren_list.append(avg_loss_siren)
            loss_speech_list.append(avg_loss_speech)
            loss_others_list.append(avg_loss_others)

        if (epoch % config.epoch_store == 0):
            current_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(current_state, config.log_indv_state_folder + "auto_encoder_model_state_" + str(epoch) + ".pt")

    torch.save(model.state_dict(), config.log_final_state_folder + "auto_encoder_model_state.pt")
    torch.save(model, config.log_complete_model_folder + "auto_encoder_model.pt")

    np.save(config.log_loss_folder + "avg_loss.npy", loss_list)
    np.save(config.log_loss_folder + "avg_loss_baby_cry.npy", loss_baby_cry_list)
    np.save(config.log_loss_folder + "avg_loss_dog.npy", loss_dog_list)
    np.save(config.log_loss_folder + "avg_loss_siren.npy", loss_siren_list)
    np.save(config.log_loss_folder + "avg_loss_speech.npy", loss_speech_list)
    np.save(config.log_loss_folder + "avg_loss_others.npy", loss_others_list)

def evaluate(model, criterion, model_state, dev_set):
    model.eval()

    sdr_list = []
    isr_list = []
    sir_list = []
    sar_list = []

    dev_set_count = 0
    loss = 0

    avg_loss = 0
    avg_loss_baby_cry = 0
    avg_loss_dog = 0
    avg_loss_siren = 0
    avg_loss_speech = 0
    avg_loss_others = 0

    state = torch.load(model_state)
    model.load_state_dict(state['state_dict'])

    dev_batch_size = 1

    if config.dataset_mode == 'all':
        dev_batch_size = len(glob.glob1(config.project_folder,"dev_set_b*"))

    for batch in range(dev_batch_size):
        if config.dataset_mode == 'all':
            dev_set = np.load(config.project_folder+"dev_set_b"+str(batch)+".npy", allow_pickle=True)

        for (norm_data, norm_target) in dev_set:
            eval_data = norm_data[0][:, :, :, :1].permute(0, 3, 2, 1)
            eval_data = eval_data.to(DEVICE)

            model_out = model(eval_data) + EPS

            model_baby_cry = model_out[:, :1, :, :]
            model_dog = model_out[:, 1:2, :, :]
            model_siren = model_out[:, 2:3, :, :]
            model_speech = model_out[:, 3:4, :, :]
            model_others = model_out[:, 4:5, :, :]

            sum_model = model_baby_cry + model_dog + model_siren + model_speech + model_others

            mask_baby_cry = model_baby_cry / sum_model
            mask_dog = model_dog / sum_model
            mask_siren = model_siren / sum_model
            mask_speech = model_speech / sum_model
            mask_others = model_others / sum_model

            calc_baby_cry = mask_baby_cry * eval_data
            calc_dog = mask_dog * eval_data
            calc_siren = mask_siren * eval_data
            calc_speech = mask_speech * eval_data
            calc_others = mask_others * eval_data

            calc_model_out = torch.cat((calc_baby_cry, calc_dog, calc_siren, calc_speech, calc_others), 1)
            calc_model_out = calc_model_out.permute(0, 3, 2, 1).to(DEVICE)

            eval_target = []
            post_eval_target = []
            model_estimate = []

            data_phase = norm_data[0][:, :, :, 1:].to(DEVICE)

            for i, element in enumerate(norm_target):
                model_out_i = torch.cat([calc_model_out[:, :, :, i:i+1], data_phase], 3).to("cpu")
                model_out_i = io_processing.post_process(model_out_i, norm_data[1], config.processing_mode)
                model_estimate.append(model_out_i)
                target_i = io_processing.post_process(norm_target[i][0], norm_target[i][1], config.processing_mode)
                post_eval_target.append(target_i)
                eval_target.append(norm_target[i][0][:, :, :, :1].permute(0, 3, 2, 1))

            concat_target = torch.cat((eval_target[0], eval_target[1], eval_target[2], eval_target[3], eval_target[4]), 1)
            concat_target = concat_target.to(DEVICE)

            concat_post_target = torch.cat((post_eval_target[0], post_eval_target[1], post_eval_target[2], post_eval_target[3], post_eval_target[4]), 0)
            concat_post_target = concat_post_target.detach().numpy()
            
            concat_estimate = torch.cat((model_estimate[0], model_estimate[1], model_estimate[2], model_estimate[3], model_estimate[4]), 0)
            concat_estimate = concat_estimate.detach().numpy()
            
            [sdr, isr, sir, sar, _] = mir_eval.separation.bss_eval_images(concat_post_target + EPS, concat_estimate + EPS)

            sdr_list.append(sdr)
            isr_list.append(isr)
            sir_list.append(sir)
            sar_list.append(sar)

            (loss_baby_cry, loss_dog, loss_siren, loss_speech, loss_others, loss_other_baby_cry, loss_other_dog, loss_other_siren, loss_other_speech, loss_other_others) = compute_loss(eval_data, concat_target, criterion, model_out)
            loss = abs(loss_baby_cry + loss_dog + loss_siren + loss_speech + loss_others - loss_other_baby_cry - loss_other_dog - loss_other_siren - loss_other_speech - loss_other_others)

            avg_loss += loss
            avg_loss_baby_cry += loss_baby_cry
            avg_loss_dog += loss_dog
            avg_loss_siren += loss_siren
            avg_loss_speech += loss_speech
            avg_loss_others += loss_others

            dev_set_count += 1

        if config.dataset_mode == 'all':
            del(dev_set)
            gc.collect()

    avg_loss /= dev_set_count
    avg_loss_baby_cry /= dev_set_count
    avg_loss_dog /= dev_set_count
    avg_loss_siren /= dev_set_count
    avg_loss_speech /= dev_set_count
    avg_loss_others /= dev_set_count

    print('Avg. Loss: {:.2f} \tAvg. Baby Loss: {:.2f} \tAvg. Dog Loss: {:.2f} \tAvg. Siren Loss: {:.2f} \tAvg. Speech Loss: {:.2f} \tAvg. Others Loss: {:.2f}'.format(
        avg_loss, avg_loss_baby_cry, avg_loss_dog, avg_loss_siren, avg_loss_speech, avg_loss_others
    ))

    np.save(config.log_bss_eval_folder+"sdr_list.npy", sdr_list)
    np.save(config.log_bss_eval_folder+"isr_list.npy", isr_list)
    np.save(config.log_bss_eval_folder+"sir_list.npy", sir_list)
    np.save(config.log_bss_eval_folder+"sar_list.npy", sar_list)

def predict(model, model_state, input_audio):
    model.eval()

    state = torch.load(model_state)
    model.load_state_dict(state['state_dict'])

    input_waveform, _ = librosa.load(input_audio, sr=config.sample_rate, mono=True)
    loop_range = 1

    if len(input_waveform) > config.waveform_duration * config.sample_rate:
        loop_range = math.ceil(len(input_waveform) / (config.waveform_duration * config.sample_rate))

    input_waveform_padded = np.zeros([loop_range * config.waveform_duration * config.sample_rate])
    input_waveform_padded[:len(input_waveform)] = input_waveform

    model_estimate = [np.empty([1, 0]) for i in range(config.num_classes)]

    for loop_index in range(loop_range):
        input_waveform_loop = input_waveform_padded[loop_index * (config.waveform_duration * config.sample_rate):(loop_index + 1) * config.waveform_duration * config.sample_rate]
        input_waveform_loop = np.expand_dims(input_waveform_loop, axis=0)
        norm_input_waveform = io_processing.normalize(io_processing.pre_process(input_waveform_loop, config.processing_mode), None, config.norm_mode)

        data = norm_input_waveform[0][:, :, :, :1].permute(0, 3, 2, 1)
        data = data.to(DEVICE)
        model_out = model(data) + EPS

        model_baby_cry = model_out[:, :1, :, :]
        model_dog = model_out[:, 1:2, :, :]
        model_siren = model_out[:, 2:3, :, :]
        model_speech = model_out[:, 3:4, :, :]
        model_others = model_out[:, 4:5, :, :]

        sum_model = model_baby_cry + model_dog + model_siren + model_speech + model_others

        mask_baby_cry = model_baby_cry / sum_model
        mask_dog = model_dog / sum_model
        mask_siren = model_siren / sum_model
        mask_speech = model_speech / sum_model
        mask_others = model_others / sum_model

        calc_baby_cry = mask_baby_cry * data
        calc_dog = mask_dog * data
        calc_siren = mask_siren * data
        calc_speech = mask_speech * data
        calc_others = mask_others * data

        calc_model_out = torch.cat((calc_baby_cry, calc_dog, calc_siren, calc_speech, calc_others), 1)
        calc_model_out = calc_model_out.permute(0, 3, 2, 1).to(DEVICE)

        for j in range(config.num_classes):
            model_out_j = torch.cat([calc_model_out[:, :, :, j:j+1], norm_input_waveform[0][:, :, :, 1:].to(DEVICE)], 3).to("cpu")
            model_out_j = io_processing.post_process(model_out_j, norm_input_waveform[1], config.processing_mode)
            model_out_j = model_out_j.detach().numpy()
            if model_estimate[j].shape[-1] + model_out_j.shape[1] <= len(input_waveform):
                model_estimate[j] = np.append(model_estimate[j], model_out_j)
            elif model_estimate[j].shape[-1] < len(input_waveform):
                model_estimate[j] = np.append(model_estimate[j], model_out_j[0, :len(input_waveform) - model_estimate[j].shape[-1]])

    for j in range(config.num_classes):
        sf.write(str(j)+"_predicted.wav", model_estimate[j].T, config.sample_rate, format='WAV', subtype='PCM_24')

def init_log():
    if not os.path.isdir(config.log_folder):
        os.mkdir(config.log_folder)
    if not os.path.isdir(config.log_final_state_folder):
        os.mkdir(config.log_final_state_folder)
    if not os.path.isdir(config.log_indv_state_folder):
        os.mkdir(config.log_indv_state_folder)
    if not os.path.isdir(config.log_complete_model_folder):
        os.mkdir(config.log_complete_model_folder)
    if not os.path.isdir(config.log_loss_folder):
        os.mkdir(config.log_loss_folder)
    if not os.path.isdir(config.log_bss_eval_folder):
        os.mkdir(config.log_bss_eval_folder)

if __name__ == '__main__':
    init_log()

    model = AutoEncoderModel()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    #optimizer = optim.Adadelta(model.parameters(), rho=0.95, lr=1)
    criterion = nn.MSELoss(reduction='sum')

    train_set = None
    dev_set = None

    invalid_input = 1

    parser = argparse.ArgumentParser(description='Autoenconder Model preparation, training, evaluation, and prediction')
    parser.add_argument('--prepdata', action='store_true',
                        help='prepares the autoencoder train and dev datasets')
    parser.add_argument('--train', action='store_true',
                        help='trains the autoencoder model')
    parser.add_argument('--eval', type=str, metavar='MODEL_STATE_PT_FILE',
                        help='evaluates the autoencoder model')
    parser.add_argument('--predict', type=str, nargs=2, metavar=('MODEL_STATE_PT_FILE', 'INPUT_FILE'),
                        help='generates a prediction through the autoencoder model')
    args = parser.parse_args()

    if(args.prepdata or args.train or args.eval):
        if os.path.exists(config.output_csv_with_audio_file_path):
            dataset_size = len(pd.read_csv(config.output_csv_with_audio_file_path))

            if args.prepdata:
                config.dataset_mode = 'all'
                io_processing.build_batch_dataset()

            if config.dataset_mode == 'indv':
                print("Initializing Autoenconder Train Dataset...")
                train_set = io_processing.AudioDataSet(0, int(config.train_set_percent * dataset_size))
                print("Initializing Autoenconder Dev Dataset...")
                dev_set = io_processing.AudioDataSet(len(train_set), dataset_size)

            if args.train:
                if train_set is None and config.dataset_mode == 'indv':
                    print("[ERROR] Train set is not defined!! \
                        Try running with --prepdata or setting config.dataset_mode to \'indv\'.")
                else:
                    print("Training the Autoencoder Model...")
                    train(model, criterion, optimizer, train_set)
            elif not(args.eval is None):
                if dev_set is None and config.dataset_mode == 'indv':
                    print("[ERROR] Dev set is not defined!! \
                        Try running with --prepdata or setting config.dataset_mode to \'indv\'.")
                elif not(os.path.exists(args.eval)):
                    print("[ERROR] MODEL_STATE_PT_FILE doesn't exist!!")
                else:
                    print("Evaluating the Autoenconder Model...")
                    evaluate(model, criterion, args.eval, dev_set)
    elif not(args.predict is None):
        print("Generating prediction through the Autoenconder Model...")
        predict(model, args.predict[0], args.predict[1])
    else:
        parser.print_help()

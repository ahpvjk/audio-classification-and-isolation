#set locations for dataset thats downloaded
MSoS_location = "./MSoS_data/"
category1_audio_file_path = MSoS_location + "Human/"
category2_audio_file_path = MSoS_location + "Nature/"
category3_audio_file_path = MSoS_location + "Urban/"
category4_audio_file_path = MSoS_location + "Speech/"
category5_audio_file_path = MSoS_location + "Effects/"
input_csv_with_audio_file_path = MSoS_location + "Logsheet_Development.csv"

path_to_english_folder = './SWC_data/'
path_to_output_folder = category4_audio_file_path
path_to_input_csv = input_csv_with_audio_file_path

#dict used for dataset generation
category_audio_file_path = {}
category_audio_file_path['Human'] = MSoS_location + "Human/"
category_audio_file_path['Nature'] = MSoS_location + "Nature/"
category_audio_file_path['Urban'] = MSoS_location + "Urban/"
category_audio_file_path['Effects'] =  MSoS_location + "Effects/"
category_audio_file_path['Speech'] =  MSoS_location + "Speech/"

# size of the dataset to be generated.
size_of_dataset = 100000
events_count = 5 # number of events to chooose from
silence_file_path=category3_audio_file_path + "silence1.wav"
max_number_of_files_others=2 # this will define the total numner files we overlay for others class

#set events to build dataset with
event_list = ["Crying baby", "Dog", "Siren", "Speaker"]

#set output folders where we can write audio files and its labels
project_folder = "./dataset/"
project_dataset_input = project_folder + "input_folder/"
project_dataset_output = project_folder + "output_folder/"
output_csv_with_audio_file_path = project_dataset_output + "labels.csv"

#set log folder where we store the model
log_folder = "./log/"
log_final_state_folder = log_folder + "final_state/"
log_indv_state_folder = log_folder + "indv_state/"
log_complete_model_folder = log_folder + "complete_model/"
log_loss_folder = log_folder + "loss/"
log_bss_eval_folder = log_folder + "bss_eval/"

#dataset parameters
dataset_mode = 'all'
train_batch_size = 35
train_set_percent = 0.93
dev_batch_size = 5
dev_set_percent = 1 - train_set_percent

#data processing parameters
sample_rate = 22050
processing_mode = 'stft'
waveform_duration = 3
stft_size = int(1024/2)
hann_window_length = stft_size
stft_freq_bins = int(stft_size/2) + 1
stft_hop = int(stft_size/4)
stft_num_frames = int(sample_rate * waveform_duration/stft_hop) + 1

#model parameters
num_classes = 5
num_inp_channels = 1
num_conv_horiz_channels = 50
num_conv_vert_channels = 30
conv_horiz_filter_size = (1, stft_freq_bins)
conv_vert_filter_size = (150, 1)
num_enc_out_features = stft_num_frames - conv_vert_filter_size[0] + 1
num_fcc_in_features = num_conv_vert_channels * num_enc_out_features
num_fcc_out_features = 128

#training parameters
num_epochs = 10000
batch_size = 1
learning_rate = 0.001
alpha_baby_cry = 0
alpha_dog = 0
alpha_siren = 0
alpha_speech = 0
alpha_others = 0

#logging parameters
epoch_log = 25

#storing parameters
epoch_store = 50

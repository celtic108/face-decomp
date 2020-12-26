phase = 4
batch_size = max(int(4), min(64, int(128/2**(phase))))
shape = (2**(phase+2), 2**(phase+2))
latent_size = 256 # Must by divisible by 16
epochs = 60000
learning_rate = 0.00001
#generated_shape = (4, 4)
save_path = './model_with_interpolation_full_dataset/sep_28_20'

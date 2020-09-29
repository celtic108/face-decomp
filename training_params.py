phase = 0
batch_size = 64//(phase+1)
shape = (2**(phase+1), 2**(phase+1))
latent_size = 512 # Must by divisible by 16
epochs = 60000
learning_rate = 0.00001
#generated_shape = (4, 4)
save_path = './model_with_interpolation_full_dataset/sep_28_20'
import torch
from my_vae_utils import ContrastiveVAE

# Set device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2
# input dim : [batch_size, 1, 160, 192, 16]
input_dim = 1 * 6 * 2 * 6  # as the example is 784(28*28 in mnist)
# ***** changed 160 to 16 for testing *****
intermediate_dim = 3
latent_dim = 2
# Instantiate the model
cvae = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, disentangle=True)
cvae = cvae.to(device)

# Create random input tensors

tg_inputs = torch.randn(batch_size, 1, 6, 2, 6).to(device)
bg_inputs = torch.randn(batch_size, 1, 6, 2, 6).to(device)
# flatten them 
tg_inputs = tg_inputs.view(batch_size, -1)
bg_inputs = bg_inputs.view(batch_size, -1)

# Test the model with generated inputs
outputs = cvae(tg_inputs, bg_inputs)

if cvae.disentangle:
    tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var,tc_loss, discriminator_loss = outputs
    print("TC Loss: ", tc_loss.mean())
    print("Discriminator Loss: ", discriminator_loss.mean())
else:
    tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var = outputs

# Print shapes of the outputs
print("tg_outputs shape: ", tg_outputs.shape)
print("bg_outputs shape: ", bg_outputs.shape)
print("tg_z_mean shape: ", tg_z_mean.shape)
print("tg_z_log_var shape: ", tg_z_log_var.shape)
print("tg_s_mean shape: ", tg_s_mean.shape)
print("tg_s_log_var shape: ", tg_s_log_var.shape)
print("bg_s_mean shape: ", bg_z_mean.shape)
print("bg_s_log_var shape: ", bg_z_log_var.shape)

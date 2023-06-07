import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from my_vae_utils import ContrastiveVAE, cvae_loss

# Define hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Instantiate the model
input_dim = 784
intermediate_dim = 256
latent_dim = 2
beta = 1
disentangle = True
gamma = 0
model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the training data
tg_train_data = ...
bg_train_data =...

# fake some data for testing
tg_train_data = torch.randn(100, 784)
bg_train_data = torch.randn(100, 784)
# Split the data into batches
tg_train_loader = torch.utils.data.DataLoader(tg_train_data, batch_size=batch_size, shuffle=True)
bg_train_loader = torch.utils.data.DataLoader(bg_train_data, batch_size=batch_size, shuffle=True)
# Loop over the data for the desired number of epochs
for epoch in range(num_epochs):
    # Loop over the batches of data
    assert len(tg_train_loader) == len(bg_train_loader) # TODO: check if this is true
    for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
        optimizer.zero_grad()
        tg_inputs, bg_inputs = batch
        # Assume disenangle == True
        output= model(tg_inputs, bg_inputs)
        tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var,tc_loss, discriminator_loss = output
        loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma)


        loss.backward()

        # Update the optimizer parameters using the gradients
        optimizer.step()

    # Print the loss for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'contrastive_vae.pth')
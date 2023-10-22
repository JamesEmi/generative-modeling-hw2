import argparse
import os
from utils import get_args

import torch
import torch.nn as nn

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    discrim_zeros = torch.zeros_like(discrim_fake)
    discrim_ones = torch.ones_like(discrim_real)
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discrim_real, discrim_ones) + criterion(discrim_fake, discrim_zeros)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    discrim_ones = torch.ones_like(discrim_fake)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discrim_fake, discrim_ones)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    # for name, param in gen.named_parameters():
    #     print(f"Parameter {name} is on {param.device}")
    
    disc = Discriminator().cuda()
        
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )

import numpy as np
import random

from avalanche.core import SupervisedPlugin

import torch
import torch.nn.functional as F

from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip


# Cusstom reservoir buffer class for samples, labels and logits
class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.stored_samples = 0

    def add(self, batch_x, batch_y, batch_logits):
        assert batch_x.size(0) == batch_y.size(0) == batch_logits.size(0)
        batch_size = batch_x.size(0)

        if self.stored_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.stored_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                samples = [(batch_x[i], batch_y[i], batch_logits[i]) for i in range(batch_size)]
                self.buffer.extend(samples)
                self.stored_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.stored_samples
                samples = [(batch_x[i], batch_y[i], batch_logits[i]) for i in range(batch_size)]
                self.buffer.extend(samples[:remaining_space])
                self.stored_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/stored_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.stored_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = (batch_x[i], batch_y[i], batch_logits[i])
            
            self.stored_samples += batch_size

    def sample(self, batch_size):
        assert batch_size <= self.stored_samples

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]

        # Stack each element of the tuples, each sample is a tuple of (x, y, logits)
        batch_x = torch.stack([sample[0] for sample in samples])
        batch_y = torch.stack([sample[1] for sample in samples])
        batch_logits = torch.stack([sample[2] for sample in samples])

        return batch_x, batch_y, batch_logits



# DER plugin class
class DerPlugin(SupervisedPlugin):
    def __init__(self, mem_size=200, alpha=0.5, beta=0.5, transform=
                 Compose([RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(p=0.5)])):
        super().__init__()
        self.buffer = ReservoirBuffer(mem_size)
        self.alpha = alpha
        self.beta = beta
        self.transform = transform

        if beta == 0:
            self.use_der_plus = False
        else:
            self.use_der_plus = True

    def before_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        # Save untransformed input
        self.original_input = strategy.mb_x.detach().clone()

        # Apply transformation to input
        strategy.mbatch[0] = self.transform(strategy.mb_x)

        # Minibatch size
        self.batch_size = strategy.mb_x.size(0)

        if len(self.buffer.buffer) > self.batch_size:
            self.use_buffer = True
            return
        else:
            # Not enough samples stored. We don't use the buffer.
            self.use_buffer = False
     
    def after_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.use_buffer:

            # Extract first batch for logits replay from old experiences
            self.repl_batch_x, _, self.repl_batch_logits = self.buffer.sample(self.batch_size)

            if self.use_der_plus:
                # Extract second batch for DER++
                repl2_batch_x, self.repl_batch_y, _ = self.buffer.sample(self.batch_size)
                self.repl_batch_x = torch.cat((self.repl_batch_x, repl2_batch_x))

            self.repl_batch_x, self.repl_batch_y, self.repl_batch_logits = (
                self.repl_batch_x.to(strategy.device),
                self.repl_batch_y.to(strategy.device),
                self.repl_batch_logits.to(strategy.device),
            )

            # Apply transformation to replay batch
            self.repl_batch_x = self.transform(self.repl_batch_x)

            # Forward on the replay batch
            self.repl_output = strategy.model(self.repl_batch_x)

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.use_buffer:

            # Compute the DER loss on the replay batch
            der_loss = self.alpha * F.mse_loss(self.repl_output[:self.batch_size], self.repl_batch_logits)
            if self.use_der_plus:
                # Add the DER++ loss
                der_loss += self.beta * F.cross_entropy(self.repl_output[self.batch_size:], self.repl_batch_y)

            strategy.loss += der_loss

    def after_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        # Add the current batch to the buffer
       self.buffer.add(self.original_input.detach(), strategy.mb_y.detach(), strategy.mb_output.detach())


    # Get the unique labels and their counts in the buffer
    def get_buffer_labels(self):
        labels_list = []
        for sample in self.buffer.buffer:
            labels_list.append(sample[1].cpu().numpy())
        # Get unique elements and their counts
        unique_elements, counts = np.unique(labels_list, return_counts=True)
        return unique_elements, counts

import numpy as np

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer 
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import torch

# Use it instead of "iter(Dataloader)" because iter is not an infinite generator 
# "iter()" raises StopIteration error when all buffer samples are used
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class DerPlugin(SupervisedPlugin):
    def __init__(self, mem_size=200, replay_mb_size=32, alpha=0.5, beta=0.5):
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)
        self.alpha = alpha
        self.beta = beta
        self.replay_mb_size = replay_mb_size

        if beta == 0:
            self.use_der_plus = False
        else:
            self.use_der_plus = True


    def before_training_exp(self, strategy: "SupervisedTemplate", num_workers: int = 0, **kwargs):
        self.exp_x, self.exp_y, self.exp_logits = [], [], []

        if len(self.buffer.buffer) == 0:
            # First experience. We don't use the buffer.
            self.first_experience = True
            return
        else:
            self.first_experience = False

        # We create a dataloader from the buffer to sample the old experiences
        self.old_exp_dataloader = cycle(
            DataLoader(self.buffer.buffer, batch_size=self.replay_mb_size, shuffle=True, drop_last=True)   
        ) 

        
    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):     
        #print('MB size: ', len(strategy.mbatch))
        if self.first_experience:
            self.original_batch_size = strategy.train_mb_size
            # No replay for first experience, no need to change the batch
            return

        # We concatenate the current batch with the batch from the buffer in order to do the forward pass.
        # 2 batches are sampled from memory for DER++ (first batch for logits replay, second batch for DER++)

        # Save the number of samples in the original (no replay) batch
        self.original_batch_size = strategy.mbatch[0].size()[0]

        # Extract first batch for logits replay from old experiences
        batch_x, _, batch_logits = next(self.old_exp_dataloader)
        batch_x, batch_logits = (
            batch_x.to(strategy.device),
            batch_logits.to(strategy.device),
        )
        # Use only inputs and logits for first replay batch
        strategy.mbatch[0] = torch.cat((strategy.mbatch[0], batch_x))
        self.batch_logits = batch_logits

        if self.use_der_plus:
            # Extract second batch for DER++
            batch_x, batch_y, _ = next(self.old_exp_dataloader)
            batch_x, batch_y = (
                batch_x.to(strategy.device),
                batch_y.to(strategy.device)
            )
            # Use only inputs and labels for second replay batch
            strategy.mbatch[0] = torch.cat((strategy.mbatch[0], batch_x))
            strategy.mbatch[1] = torch.cat((strategy.mbatch[1], batch_y))
        
        # print sizes of the batch
        print('X size: ', strategy.mbatch[0].size())
        print('Y size: ', strategy.mbatch[1].size())
        print('Logits size: ', self.batch_logits.size())


        
    def after_forward(self, strategy: "SupervisedTemplate", **kwargs):
        # Store the not replayed part of the batch (current experience) for buffer update
        # Save X, Y and logits
        new_exp_betch = AvalancheDataset(TensorDataset(
            strategy.mb_x[:self.original_batch_size],
            strategy.mb_y[:self.original_batch_size],
            strategy.mb_output[:self.original_batch_size]))

        self.buffer.update_from_dataset(new_exp_betch, **kwargs)

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        # We set the default criterion to 0 to avoid computing the loss twice
        def void_criterion(*args, **kwargs):
            return 0
        strategy.criterion = void_criterion

        if self.first_experience:
            # No replay for first experience, no need to use DER++ loss
            strategy.loss =  F.cross_entropy(
                strategy.mb_output[:self.original_batch_size],
                strategy.mb_y[:self.original_batch_size],
            )
        elif self.use_der_plus:
            # After first experience. We change the loss function to include DER++ loss
            strategy.loss =  F.cross_entropy(
                strategy.mb_output[:self.original_batch_size],
                strategy.mb_y[:self.original_batch_size],
            )
            + self.alpha * F.mse_loss(
                strategy.mb_output[self.original_batch_size:self.original_batch_size+self.replay_mb_size],
                self.batch_logits,
            ) 
            + self.beta * F.cross_entropy(
                strategy.mb_output[self.original_batch_size+self.replay_mb_size:],
                strategy.mb_y[self.original_batch_size:],
            )
        else:
            # After first experience. Only use DER loss
            strategy.loss =  F.cross_entropy(
                strategy.mb_output[:self.original_batch_size],
                strategy.mb_y[:self.original_batch_size],
            )
            + self.alpha * F.mse_loss(
                strategy.mb_output[self.original_batch_size:],
                self.batch_logits,
            )


    def after_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if not self.first_experience:
            # Restore the original batch (no replay)
            strategy.mbatch[0] = strategy.mbatch[0][:self.original_batch_size]
            strategy.mbatch[1] = strategy.mbatch[1][:self.original_batch_size]

            # Also cut the output to the original batch size
            strategy.mb_output = strategy.mb_output[:self.original_batch_size]

    # def after_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        # print sizes of the batch 
        # print('original_batch_size: ', self.original_batch_size)
        # print('X size: ', strategy.mbatch[0].size())
        # print('Y size: ', strategy.mbatch[1].size())
        # print("mb_output size: ", strategy.mb_output.size())


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # print the number of unique labels in the buffer
        print('Unique y in buffer: ', (self.buffer.buffer))
        labels_list = []
        for sample in self.buffer.buffer:
            labels_list.append(sample[1].cpu().numpy())
        # Get unique elements and their counts
        unique_elements, counts = np.unique(labels_list, return_counts=True)

        # Print unique elements and their counts
        for element, count in zip(unique_elements, counts):
            print(f"Element: {element}, Count: {count}")

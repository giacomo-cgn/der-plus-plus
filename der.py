from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer 
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import torch

# Use it instead of "iter(cycle)" because iter is not an infinite generator 
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

        self.first_epoch = True

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

        print('BUFFER SIZE: ', len(self.buffer.buffer))
        for el in self.buffer.buffer:
            print('EL SIZE: ', el[0].shape, el[1].shape, el[2].shape)

        
    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):     
        #print('MB size: ', len(strategy.mbatch))
        if self.first_experience:
            # No replay for first experience, no need to change the batch
            return

        # We concatenate the current batch with the batch from the buffer in order to do the forward pass.
        # 2 batches are used for DER++ (first batch for logits replay, second batch for DER++)

        # Save the orignal batch (without replay)
        self.original_batch = strategy.mbatch.copy()

        # Extract first batch for logits replay from old experiences
        batch_x, _, batch_logits = next(self.old_exp_dataloader)
        batch_x, batch_logits = (
            batch_x.to(strategy.device),
            batch_logits.to(strategy.device),
        )
        # Use only inputs and logits for first replay batch
        strategy.batch_logits = batch_logits
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

        
    def after_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.first_epoch:
            # Only for first epoch because data is the same for all epochs
            # Store the batch (X and Y and logits) of the current experience for future buffer update
            self.exp_x.append(strategy.mb_x[:strategy.train_mb_size])
            self.exp_y.append(strategy.mb_y[:strategy.train_mb_size])
            self.exp_logits.append(strategy.mb_output[:strategy.train_mb_size])


    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        # We set the default criterion to 0 to avoid computing the loss twice
        def void_criterion(*args, **kwargs):
            return 0
        strategy.criterion = void_criterion

        if self.first_experience:
            # No replay for first experience, no need to use DER++ loss
            strategy.loss =  F.cross_entropy(
                strategy.mb_output[:strategy.train_mb_size],
                strategy.mb_y[:strategy.train_mb_size],
            )
        else:
            # After first experience. We change the loss function to include DER++ loss
            strategy.loss =  F.cross_entropy(
                strategy.mb_output[:strategy.train_mb_size],
                strategy.mb_y[:strategy.train_mb_size],
            )
            + self.alpha * F.mse_loss(
                strategy.mb_output[strategy.train_mb_size:strategy.train_mb_size+self.replay_mb_size],
                strategy.batch_logits,
            ) 
            + self.beta * F.cross_entropy(
                strategy.mb_output[strategy.train_mb_size+self.replay_mb_size:],
                strategy.mb_y[strategy.train_mb_size:],
            )


    def after_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if not self.first_experience:
            # Restore the original batch (no replay)
            strategy.mbatch = self.original_batch
            # Also cut the output to the original batch size
            strategy.mb_output = strategy.mb_output[:strategy.train_mb_size]


    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        self.first_epoch = False


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # Generate a dataset from the collected X, Y and logits of the current experience
        self.exp_x = torch.cat(self.exp_x, dim=0)
        self.exp_y = torch.cat(self.exp_y, dim=0)
        self.exp_logits = torch.cat(self.exp_logits, dim=0)
        self.exp_dataset = AvalancheDataset(TensorDataset(self.exp_x, self.exp_y, self.exp_logits))

        # Update the memory buffer with the current experience
        self.buffer.update_from_dataset(self.exp_dataset, **kwargs)
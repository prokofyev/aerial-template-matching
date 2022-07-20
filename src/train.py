from typing import Optional, List
import sys
import gc
from collections import namedtuple
import torch
from tqdm import tqdm

from .utils import RunningAverage, compute_metric

if __name__ == '__main__':
    pass

EpochStats = namedtuple('EpochStats', 'epoch learning_rate train_loss val_loss val_metric time')

MSE = torch.nn.MSELoss(reduction='sum')


def criterion(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Custom criterion for competition

    Parameters
    ----------
    y_pred (torch.Tensor): predicted labels
    y_true (torch.Tensor): true labels

    Returns
    ----------
    criterion (torch.Tensor): calculated criterion
    """
    batch_size = y_pred.shape[0]
    alpha = 16
    loss = 0.7 * MSE(y_pred[:, :4], y_true[:, :4]) + 0.3 * torch.min(MSE(y_pred[:, 4], y_true[:, 4]),
                                                                     MSE(1 + y_pred[:, 4], y_true[:, 4]))
    loss /= batch_size
    return alpha * loss


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          epochs: int,
          lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
          filename: str,
          accumulate_every_n_epochs: int = 1,
          clip_gradient: bool = False) -> List[EpochStats]:
    """
    Train the model and evaluate every epoch

    Parameters
    ----------
    model (torch.nn.Module): pytorch neural network model
    optimizer (torch.optim.Optimizer): pytorch optimizer object
    criterion (torch.nn.Module): pytorch criterion that computes a gradient according to a given loss function
    train_loader (torch.utils.data.DataLoader): pytorch data loading iterable over the training dataset
    val_loader (torch.utils.data.DataLoader): pytorch data loading iterable over the validation dataset
    epochs (int): total number of epochs
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
    filename (str): string containing the filename to save the model and optimizer states to a disk file
    accumulate_every_n_epochs (int): epochs to accumulate gradient before updating the weights
    clip_gradient (bool): if True, will clip the gradient using gradient norm

    Returns
    -------
    history (List[EpochStats]): training history
    """
    device = torch.device('cuda')
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    history = []
    best_metric = 0.0
    denorm = torch.tensor([10496, 10496, 10496, 10496, 360]).to(device)

    for e in range(epochs):
        loss_avg = RunningAverage()
        val_metric_avg = RunningAverage()
        val_loss_avg = RunningAverage()

        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        with tqdm(total=len(train_loader), leave=False, file=sys.stdout) as t:
            stats_current_lr = optimizer.param_groups[0]['lr']
            t.set_description(f'Epoch {e + 1}, LR {stats_current_lr:.6f}')

            for batch_n, batch_data in enumerate(train_loader):
                train_batch, labels_batch = batch_data
                train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)

                with torch.autocast(device_type='cuda'):
                    output_batch = model(train_batch)
                    loss = criterion(output_batch, labels_batch)
                    loss_avg.update(loss.item())
                    loss /= accumulate_every_n_epochs

                scaler.scale(loss).backward()

                if (batch_n + 1) % accumulate_every_n_epochs == 0:
                    if clip_gradient:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                t.set_postfix({'stats': f'train_loss: {loss_avg():.4f}'})
                t.update()
                stats_time_elapsed = t.format_interval(t.format_dict['elapsed'])

        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                val_batch, val_labels_batch = batch_data
                val_batch, val_labels_batch = val_batch.to(device), val_labels_batch.to(device)
                val_output_batch = model(val_batch)

                val_loss = criterion(val_output_batch, val_labels_batch)
                val_loss_avg.update(val_loss.item())
                val_metric_batch = compute_metric((val_labels_batch.squeeze(0) * denorm).cpu().numpy(),
                                                  (val_output_batch.squeeze(0) * denorm).cpu().numpy())
                val_metric_avg.update(val_metric_batch.item())

        stats_epoch = EpochStats(epoch=e + 1,
                                 learning_rate=stats_current_lr,
                                 train_loss=loss_avg(),
                                 val_loss=val_loss_avg(),
                                 val_metric=val_metric_avg(),
                                 time=stats_time_elapsed)
        history.append(stats_epoch)

        print(
            f'Epoch {stats_epoch.epoch}. LR {stats_epoch.learning_rate:.6f}, train_loss: {stats_epoch.train_loss:.4f},'
            f' val_loss: {stats_epoch.val_loss:.4f}, val_metric: {stats_epoch.val_metric:.4f}, time: {stats_epoch.time}')

        if val_metric_avg() > best_metric:
            best_metric = val_metric_avg()
            torch.save(model, filename)

    return history

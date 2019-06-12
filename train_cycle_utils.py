import os
import torch
from torch.utils.data import DataLoader
from data_loader_utils import MelSpectrogramDataset
import numpy as np
from tensorboardX import SummaryWriter


DEFAULT_SCHED_PARAM_DICT = dict()


def make_checkpoint(path, model, optimizer, lr, scheduler, iter, verbose=True):
    """Saves intermediate state of the training procedure into file.

    Parameters
    ----------
    path : str
        Path to save parameters.
    model : nn.Module
        Trained model.
    optimizer : Pytorch Optimizer
        Model optimizer.
    lr : float > 0 [scalar]
        Optimizer's learning rate.
    scheduler : Pytorch LRScheduler
        Optimizer's learning rate scheduler.
    iter : int >= 0 [scalar]
        Number of the current iteration.
    verbose : bool, default -- True
        Whether print auxiliary output.

    """
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr,
                'scheduler': scheduler if scheduler is None else scheduler.state_dict(),
                'iter': iter
                }, path)

    if verbose:
        print("Saved model state at iteration {} as {}".format(iter, path))


def load_checkpoint(path, model, optimizer, scheduler, verbose=True):
    """Loads intermediate state of the training procedure from file.

    Parameters
    ----------
    path : str
        Path to saved parameters.
    model : nn.Module
        Trained model.
    optimizer : Pytorch Optimizer
        Model optimizer.
    scheduler : Pytorch LRScheduler
        Optimizer's learning rate scheduler.
    verbose : bool, default -- True
        Whether print auxiliary output.

    Returns
    -------
    model : nn.Module
        Updated trained model.
    optimizer : Pytorch Optimizer
        Updated model optimizer.
    scheduler : Pytorch LRScheduler
        Updated optimizer's learning rate scheduler.
    iter : int >= 0 [scalar]
        Number of the current iteration.
    lr : float > 0 [scalar]
        Updated optimizer's learning rate.

    """
    assert os.path.isfile(path)

    param_dict = torch.load(path, map_location='cpu')

    model.load_state_dict(param_dict['model'])
    optimizer.load_state_dict(param_dict['optimizer'])
    lr = param_dict['lr']

    if (param_dict['scheduler'] is None) or (scheduler is None):
        scheduler = None
    else:
        scheduler.load_state_dict(param_dict['scheduler'])

    iter = param_dict['iter']

    if verbose:
        print("Loaded checkpoint {} at iteration {}" .format(path, iter))
    return model, optimizer, scheduler, iter, lr


def train_cycle(model, model_name, save_dir, criterion, dataset_params, n_epochs, batch_size=3,
                optimizer=None, lr=1e-4, scheduler=None, scheduler_state_dict=DEFAULT_SCHED_PARAM_DICT,
                device='cpu', seed=42, checkpoint_path=None, iter_checkpoint_hop=2000, verbose=True,
                log_dir=None, exp_smooth_val=0.5):
    """Main training cycle.

    Parameters
    ----------
    model : nn.Module
         Model to train.
    model_name : str
        Model tag.
    save_dir : str
        Path to the directory with saved model's checkpoints.
    criterion : callable
        Optimization criterion.
    dataset_params : dict
        Dictionary with parameters of the MelSpectrogramDataset class. Defines following parameters:
        `data_dir`, `sr`, `n_fft`, `fmin`, `fmax`, `hop_len`, `win_len`, `seg_len`, `seed`, `shuffle`,
        max_wav_val`.
    n_epochs : int > 0 [scalar]
        Number of the training epochs.
    batch_size : int > 0 [scalar], default -- 3
        Size of the batch.
    optimizer : Pytorch Optimizer, default -- None
        Model optimizer. If optimizer == None, then Adam optimizer is used.
    lr : float > 0 [scalar], default -- 1e-4
        Learning rate.
    scheduler : Pytorch LRScheduler, default -- None
        Optimizer's learning rate scheduler. If scheduler == None, then no scheduling.
    scheduler_state_dict : dict, default -- dict()
        Pytorch LRScheduler parameters dictionary.
    device : str, default -- 'cpu'
        Name of the device where the model is trained. Possible values: 'cpu', 'cuda'.
    seed : int [scalar]
        Seed for pseudo random numbers generator.
    checkpoint_path : str, default -- None
        Path to initial state of the model. If checkpoint_path == None, then
        initial state won't be downloaded from the file.
    iter_checkpoint_hop : int > 0 [scalar], default -- 2000
        Period length between savings of the model's states.
    verbose : bool, default -- True
        Whether print auxiliary messages to stdout.
    log_dir : str, default -- None
        Path to the directory to store TensorBoard logs.
        If log_dir == None,then TensorBoard won't be used.
    exp_smooth_val : float between 0 and 1, default -- 0.5
        Exponential smoothing parameter for epoch loss and gradient norm estimation.

    """
    if not (seed is None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device != 'cpu':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    dataset = MelSpectrogramDataset(**dataset_params)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, drop_last=True)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.chmod(save_dir, 0o777)
        if verbose:
            print("Created save directory", save_dir)

    if not (log_dir is None):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            os.chmod(log_dir, 0o777)
            if verbose:
                print("Created log directory", log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not (scheduler is None):
        scheduler = scheduler(optimizer, **scheduler_state_dict)

    iter = 0
    if not (checkpoint_path is None):
        model, optimizer, scheduler, iter, lr = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=verbose)
        iter += 1

    model.train()
    for epoch in range(iter // len(train_loader), n_epochs):
        if verbose:
            print("Started epoch: {}".format(epoch))

        if not (log_dir is None):
            epoch_norm, epoch_loss = 0.0, 0.0

        for mel, audio in train_loader:
            mel, audio = mel.to(device), audio.to(device)

            model.zero_grad()

            out = model(audio, mel)

            loss = criterion(out)
            loss.backward()

            optimizer.step()

            if verbose:
                print("    iteration {}:\t{:.8f}".format(iter, loss.item()))

            if iter % iter_checkpoint_hop == 0:
                save_path = "{}/{}_{}.ckpt".format(save_dir, model_name, iter)
                make_checkpoint(path=save_path,
                                model=model,
                                optimizer=optimizer,
                                lr=lr,
                                scheduler=scheduler,
                                iter=iter,
                                verbose=verbose)

            if not (log_dir is None):
                total_norm = 0.0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm **= 0.5

                epoch_norm = (1 - exp_smooth_val) * epoch_norm + exp_smooth_val * total_norm
                epoch_loss = (1 - exp_smooth_val) * epoch_loss + exp_smooth_val * loss.item()

                writer.add_scalar('{}/iter_grad_norm'.format(model_name), total_norm, iter)
                writer.add_scalar('{}/iter_loss'.format(model_name), loss.item(), iter)

            iter += 1

        if not (scheduler is None):
            scheduler.step()

        if not (log_dir is None):
            writer.add_scalar('{}/epoch_grad_norm'.format(model_name), epoch_norm, epoch)
            writer.add_scalar('{}/epoch_loss'.format(model_name), epoch_loss, epoch)

    model.eval()
    model.cpu()

    save_path = "{}/{}_final_model_state.ckpt".format(save_dir, model_name)
    torch.save(model.state_dict(), save_path)

    if not (log_dir is None):
        writer.export_scalars_to_json("{}/{}_training_information.json".format(log_dir, model_name))
        writer.close()

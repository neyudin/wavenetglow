import os
import torch
from torch.utils.data import DataLoader
from data_loader_utils import MelSpectrogramDataset
import numpy as np
from tensorboardX import SummaryWriter

from scipy.io.wavfile import write


DEFAULT_SCHED_PARAM_DICT = dict()


class VerboseStringPadding:
    """Auxiliary tool for inline string printing.

    This class helps to print all messages upon each other.

    Attributes
    ----------
    _prev_str_len : int >= 0 [scalar], default -- 0
        Length of the previous printed string.
    """
    def __init__(self):
        self._prev_str_len = 0

    def print_string(self, string):
        """Print string with whitespace padding.

        Parameters
        ----------
        string : str
            String to print.

        """
        mod_string = '\r' + string
        n_chars = len(mod_string)
        pad_len = max(self._prev_str_len - n_chars, 0)
        self._prev_str_len = n_chars
        print(mod_string + ' ' * pad_len, end='', flush=True)


def make_checkpoint(path, model, optimizer, lr, scheduler, iter, verbose=None):
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
    verbose : printing class, default -- None
        Class with method .print_string(string) to print string.

    """
    torch.save({'model': model.state_dict(),
                'model_class': model,
                'optimizer': optimizer.state_dict(),
                'lr': lr,
                'scheduler': scheduler if scheduler is None else scheduler.state_dict(),
                'iter': iter
                }, path)

    if not (verbose is None):
        verbose.print_string("Saved model state at iteration {} as {}".format(iter, path))


def load_checkpoint(path, model, optimizer, scheduler, verbose=None):
    """Loads intermediate state of the training procedure from file.

    Parameters
    ----------
    path : str
        Path to saved parameters.
    model : nn.Module
        Pytorch model.
    optimizer : Pytorch Optimizer
        Model optimizer.
    scheduler : Pytorch LRScheduler
        Optimizer's learning rate scheduler.
    verbose : printing class, default -- None
        Class with method .print_string(string) to print string.

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

    if model is None:
        model = param_dict['model_class']
        model.load_state_dict(param_dict['model'])
    else:
        model.load_state_dict(param_dict['model'])
    optimizer.load_state_dict(param_dict['optimizer'])
    lr = param_dict['lr']

    if (param_dict['scheduler'] is None) or (scheduler is None):
        scheduler = None
    else:
        scheduler.load_state_dict(param_dict['scheduler'])

    iter = param_dict['iter']

    if not (verbose is None):
        verbose.print_string("Loaded checkpoint {} at iteration {}" .format(path, iter))
    return model, optimizer, scheduler, iter, lr


def train_cycle(model, model_name, save_dir, criterion, dataset_params, val_dataset_params, n_epochs,
                val_sigma=0.6, stop_iter_num=580000, batch_size=24, sigma=1.0, optimizer=None, lr=1e-4,
                scheduler=None, scheduler_state_dict=DEFAULT_SCHED_PARAM_DICT, max_norm=None,
                device='cuda', seed=42, checkpoint_path=None, iter_checkpoint_hop=2000,
                verbose=None, batch_verbose_period=100, log_dir=None, exp_smooth_val=0.4):
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
        Dictionary with parameters of the MelSpectrogramDataset class for training. Defines following parameters:
        `data_dir`, `sr`, `n_fft`, `fmin`, `fmax`, `hop_len`, `win_len`, `seg_len`, `seed`, `shuffle`,
        `max_wav_val`.
    val_dataset_params: dict
        Dictionary with parameters of the MelSpectrogramDataset class for validation. Defines following parameters:
        `data_dir`, `sr`, `n_fft`, `fmin`, `fmax`, `hop_len`, `win_len`, `seg_len`, `seed`, `shuffle`,
        `max_wav_val`.
    n_epochs : int > 0 [scalar]
        Number of the training epochs.
    val_sigma : float > 0 [scalar], default -- 0.6
        Standard deviation for waveglow inference.
    stop_iter_num : int > 0 [scalar], default -- 580000
        Maximal number of the iteration.
    batch_size : int > 0 [scalar], default -- 24
        Size of the batch.
    sigma : float > 0 [scalar], default -- 1.0
        Standard deviation for Normal distribution.
    optimizer : Pytorch Optimizer, default -- None
        Model optimizer. If optimizer == None, then Adam optimizer is used.
    lr : float > 0 [scalar], default -- 1e-4
        Learning rate.
    scheduler : Pytorch LRScheduler, default -- None
        Optimizer's learning rate scheduler class reference. If scheduler == None, then no scheduling.
    scheduler_state_dict : dict, default -- dict()
        Pytorch LRScheduler parameters dictionary.
    max_norm : float > 0, [scalar], default -- None
        Maximal value of gradient norm. If max_norm != None, then gradient clipping is performed.
    device : str, default -- 'cuda'
        Name of the device where the model is trained. Possible values: 'cpu', 'cuda'.
    seed : int [scalar]
        Seed for pseudo random numbers generator.
    checkpoint_path : str, default -- None
        Path to initial state of the model. If checkpoint_path == None, then
        initial state won't be downloaded from the file.
    iter_checkpoint_hop : int > 0 [scalar], default -- 2000
        Period length between savings of the model's states.
    verbose : printing class, default -- None
        Class with method .print_string(string) to print string.
    batch_verbose_period : int, default -- 100
        Period length between printing model's loss.
    log_dir : str, default -- None
        Path to the directory to store TensorBoard logs.
        If log_dir == None, then TensorBoard won't be used.
    exp_smooth_val : float between 0 and 1, default -- 0.4
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
    val_dataset = MelSpectrogramDataset(**val_dataset_params)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.chmod(save_dir, 0o777)
        if not (verbose is None):
            verbose.print_string("Created save directory {}".format(save_dir))

    samples_dir = save_dir + "/samples/"
    if not os.path.isdir(samples_dir):
        os.makedirs(samples_dir)
        os.chmod(samples_dir, 0o777)
        if not (verbose is None):
            verbose.print_string("Created samples directory {}".format(samples_dir))

    if not (log_dir is None):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            os.chmod(log_dir, 0o777)
            if not (verbose is None):
                verbose.print_string("Created log directory {}".format(log_dir))
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

    epoch_norm, epoch_loss = 0.0, 0.0

    for epoch in range(iter // len(train_loader), n_epochs):
        if not (verbose is None):
            verbose.print_string("Started epoch: {}".format(epoch))

        model.train()

        for i, (mel, audio) in enumerate(train_loader):
            if iter == stop_iter_num:
                break

            mel, audio = mel.to(device), audio.to(device)

            model.zero_grad()

            out = model(audio, mel)

            if criterion is None:
                loss = model.compute_loss(audio, mel, sigma=sigma)
            else:
                loss = criterion(out)

            loss.backward()

            if not (max_norm is None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            if not (verbose is None):
                if iter % batch_verbose_period == 0:
                    verbose.print_string("iteration {} ({} of {} in epoch), loss: {:.8f}".format(
                        iter, i, len(train_loader), loss.item()))

            if iter % iter_checkpoint_hop == 0:
                save_path = "{}/{}_{}.ckpt".format(save_dir, model_name, iter)
                make_checkpoint(path=save_path,
                                model=model,
                                optimizer=optimizer,
                                lr=lr,
                                scheduler=scheduler,
                                iter=iter,
                                verbose=verbose)

            total_norm = 0.0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm **= 0.5

            epoch_norm = (1 - exp_smooth_val) * epoch_norm + exp_smooth_val * total_norm
            epoch_loss = (1 - exp_smooth_val) * epoch_loss + exp_smooth_val * loss.item()

            if not (log_dir is None):
                writer.add_scalar('{}/iter_grad_norm'.format(model_name), total_norm, iter)
                writer.add_scalar('{}/iter_loss'.format(model_name), loss.item(), iter)

            iter += 1

        if not (scheduler is None):
            scheduler.step()

        if not (log_dir is None):
            writer.add_scalar('{}/epoch_grad_norm'.format(model_name), epoch_norm, epoch)
            writer.add_scalar('{}/epoch_loss'.format(model_name), epoch_loss, epoch)

        model.eval()

        epoch_samples_dir = samples_dir + "epoch_{}/".format(epoch)
        if not os.path.isdir(epoch_samples_dir):
            os.makedirs(epoch_samples_dir)
            os.chmod(epoch_samples_dir, 0o777)
            if not (verbose is None):
                verbose.print_string("Created samples directory {} for epoch {}".format(epoch_samples_dir, epoch))

        for val_iter in range(len(val_dataset)):
            mel, audio = val_dataset[val_iter]
            mel = mel.to(device).view(1, mel.shape[0], mel.shape[1])

            est_audio = model.infer(mel, sigma=val_sigma).squeeze(0)

            audio, est_audio = audio.to('cpu'), est_audio.to('cpu')

            write(epoch_samples_dir + "inference_{}.wav".format(val_iter), val_dataset_params['sr'],
                  est_audio.clamp(-1, 1).numpy())
            write(epoch_samples_dir + "ground_truth_{}.wav".format(val_iter), val_dataset_params['sr'],
                  audio.clamp(-1, 1).numpy())

        if iter == stop_iter_num:
            break

    model.cpu()

    save_path = "{}/{}_final_model_state.ckpt".format(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    save_path = "{}/{}_final_model.ckpt".format(save_dir, model_name)
    torch.save(model, save_path)

    if not (log_dir is None):
        writer.export_scalars_to_json("{}/{}_training_information.json".format(log_dir, model_name))
        writer.close()

    print(flush=True)

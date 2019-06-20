# WaveGlow
Unofficial implementation of [WaveGlow](https://arxiv.org/abs/1811.00002) model.

## Project participants

* [Gleb Maslyakov](https://github.com/glebmaslyak)

* [Sergey Nikolaev](https://github.com/corvin28)

* [Nikita Yudin](https://github.com/neyudin)

## Preliminary settings

Clone this repository and initialize submodule, install project requirements:

```
    git clone https://github.com/neyudin/wavenetglow.git
    cd wavenetglow
    git submodule init
    git submodule update
    pip install -r requirements.txt
```

## Train the model

1. Download [LJ Speech dataset](https://mega.nz/#!OPwTQKCQ!cJAcqPS9hn705MeeI6JwRGPEtz39kjjGySGv2IN4xNE) and unpack it to repository root.

2. Set training configuration in [file](./config.json).

3. Start training process:

```
    python train.py -c config.json
```

The whole training process can be monitored via Tensorboard while executing the following command (`<log_dir>` — directory with saved logs for Tensorboard, `log_dir` parameter in configuration [file](./config.json) stands for it, `<port_num>` — port number to watch training information via `http://localhost:<port_num>`):

```
    tensorboard --logdir <log_dir> --port <port_num>
```

## Text-to-Speech inference with pretrained Tacotron2 model

1. Train WaveGlow model from scratch on [data](https://mega.nz/#!OPwTQKCQ!cJAcqPS9hn705MeeI6JwRGPEtz39kjjGySGv2IN4xNE).

2. Download pretrained Tacotron2 [model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view).

3. Run inference demo in [notebook](./inference.ipynb).

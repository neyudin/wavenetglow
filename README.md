# wavenetglow
Основной репозиторий по проекту в рамках курса "Современные методы распознавания и синтеза речи". Реализация [WaveGlow](https://arxiv.org/abs/1811.00002).

## Участники проекта

* [Масляков Глеб](https://github.com/glebmaslyak)

* [Николаев Сергей](https://github.com/corvin28)

* [Юдин Никита](https://github.com/neyudin)

## Подготовка к работе

Для работы с реализацией модели WaveGlow необходимо выполнить следующие команды

```
    git clone https://github.com/neyudin/wavenetglow.git
    cd wavenetglow
    git submodule init
    git submodule update
    pip install -r requirements.txt
```

## Обучение модели

1. Загрузить [выборку](https://mega.nz/#!OPwTQKCQ!cJAcqPS9hn705MeeI6JwRGPEtz39kjjGySGv2IN4xNE) и распаковать её в корне данного репозитория.

2. Настроить конфигурацию обучения модели в [файле](./config.json).

3. Запустить процесс командой:

```
    python train.py -c config.json
```

Процесс обучения можно отслеживать с помощью Tensorboard, запустив команду (`<log_dir>` — директория, в которую сохраняются логи, параметр `log_dir` в [файле](./config.json) конфигурации, `<port_num>` — номер порта, по которому в браузере на `localhost` выводится информация о процессе обучения):

```
    tensorboard --logdir <log_dir> --port <port_num>
```

## Вывод в задаче Text-to-Speech с предобученными моделями

1. Загрузить предобученную модель по [ссылке](https://mega.nz/#!CWpAiAQZ!CZuZ0rgTttqPKe3wrmH7_Cj9Neb0bvlYMrieKTXlKkw) или обучить на [данных](https://mega.nz/#!OPwTQKCQ!cJAcqPS9hn705MeeI6JwRGPEtz39kjjGySGv2IN4xNE).

2. Загрузить предобученную модель Tacotron2 по [ссылке](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view).

3. Запустить демонстрационную версию в [ноутбуке](./inference.ipynb).

"""This module trains WaveGlow from scratch"""
import argparse
import json
import train_cycle_utils
import WaveGlow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for training configuration')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    model = WaveGlow.WaveGlow(**config['WaveGlow_params'])

    verbose = train_cycle_utils.VerboseStringPadding()

    train_cycle_utils.train_cycle(model=model, verbose=verbose, **config['train_cycle_params'])

from utils import load_config
import variants

if __name__ == '__main__':
    config = load_config()

    if config.mode == 'train':
        entry = getattr(variants.train, config.training.get('entry', 'train_dsp'))
    elif config.mode == 'infer':
        entry = getattr(variants.infer, config.inference.get('entry', 'infer_dsp'))

    entry(config)
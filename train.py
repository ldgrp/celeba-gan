from tap import Tap
from typing import Optional
from pathlib import Path
from utils import Config, set_logger
from tensorflow.data.experimental import AUTOTUNE
from model import GAN
import logging
import os
import sys
import tensorflow as tf

class Arguments(Tap):
    input_dir: Optional[str] = 'data'  # Dataset directory
    output_dir: Optional[str] = 'output'  # Output directory
    config: Optional[str] = None # Config file

def parse_tfrecord(record):
    '''
    Parse a TFRecord into an image
    '''
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)
    })

    shape = features['shape']
    data = tf.io.decode_raw(features['data'], tf.float32)
    img = tf.reshape(data, shape)
    return img

if __name__ == '__main__':
    args = Arguments().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup logging
    set_logger(output_dir / 'train.log')

    # Load configuration file
    config = Config()

    if args.config:
        print(args.config)
        config.load_json(args.config)

    strategy = tf.distribute.MirroredStrategy()

    config.global_batch_size = config.batch_size * strategy.num_replicas_in_sync

    # Create input pipeline 
    logging.info('Loading input files...')
    input_dir = Path(args.input_dir)
    files = tf.io.matching_files(f'{input_dir}/*.tfrecord')
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
        .take(128)
        .cache()
        .shuffle(config.buffer_size)
        .batch(config.global_batch_size)
        .prefetch(AUTOTUNE)
    )

    dataset = strategy.experimental_distribute_dataset(dataset)

    # Begin training
    logging.info('Training...')
    with strategy.scope():
        gan = GAN(config)
        gan.train(dataset, config.epochs)

        gan.generator.save(config.output_dir / 'models' / 'generator')
        gan.discriminator.save(config.output_dir / 'models' / 'discriminator')
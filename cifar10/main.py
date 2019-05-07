import argparse
import inspect
import math
import numpy as np
import os
import sys
import tensorflow as tf
import time

import attacks
import cifar10_input
from model import Model

""" The collection of all attack classes """
ATTACK_CLASSES = [
  x for x in attacks.__dict__.values() 
  if inspect.isclass(x)
]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" Arguments """
DATA_DIR = '/data_large/readonly/cifar10_data'
MODEL_DIR = './models/adv_trained'

parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--data_dir', default=DATA_DIR, type=str)
parser.add_argument('--model_dir', default=MODEL_DIR, type=str)
parser.add_argument('--asset_dir', default='./assets')
parser.add_argument('--save_dir', default='./saves')

# Experiment Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')

# Attack setting
parser.add_argument('--attack', default='ParsimoniousAttack', type=str, help='The type of attack')
parser.add_argument('--loss_func', default='xent', type=str, help='The type of loss function')
parser.add_argument('--epsilon', default=8, type=int, help='The maximum perturbation')
parser.add_argument('--max_queries', default=20000, type=int, help='The query limit')
parser.add_argument('--targeted', action='store_true', help='Targeted attack if true')

# Local Search setting
parser.add_argument('--max_iters', default=1, type=int, help='The number of iterations in local search')
parser.add_argument('--block_size', default=4, type=int, help='Initial block size')
parser.add_argument('--batch_size', default=64, type=int, help='The size of batch. No batch if negative')
parser.add_argument('--no_hier', action='store_true', help='No hierarchical evaluation if true')

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Load the pretrained model
  model_file = tf.train.latest_checkpoint(args.model_dir)
  if model_file is None:
    tf.logging.info('No model found')
    sys.exit()
  
  # Create a session
  sess = tf.InteractiveSession()
   
  # Build a graph
  model = Model(mode='eval')
 
  # Restore the checkpoint
  saver = tf.train.Saver()
  saver.restore(sess, model_file)
  
  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(model, args)
  
  # Create a directory
  if args.save_img:
    tf.gfile.MakeDirs(args.save_dir)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load dataset
  cifar = cifar10_input.CIFAR10Data(args.data_dir)
  
  # Load the indices
  indices = np.load(os.path.join(args.asset_dir, 'indices_untargeted.npy')) 
  
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_queries = []
  index_to_query = {}

  while count < args.sample_size:
    tf.logging.info("")

    # Get an image and the corresponding label
    initial_img = cifar.eval_data.xs[indices[index]]
    initial_img = np.int32(initial_img)
    initial_img = np.expand_dims(initial_img, axis=0)
    orig_class = cifar.eval_data.ys[indices[index]]
   
    # Generate a target class (same method as in Boundary attack)
    if args.targeted:
      target_class = (orig_class+1) % 10
      target_class = np.expand_dims(target_class, axis=0)

    orig_class = np.expand_dims(orig_class, axis=0)

    count += 1
   
    # Run attack
    if args.targeted:
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
        count, indices[index], orig_class[0], target_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, target_class, indices[index], sess)
    else:
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
        count, indices[index], orig_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, orig_class, indices[index], sess)
    
    # Check if the adversarial image satisfies the constraint 
    assert np.amax(np.abs(adv_img-initial_img)) <= args.epsilon
    assert np.amax(adv_img) <= 255
    assert np.amin(adv_img) >= 0
    p = sess.run(model.predictions, feed_dict={model.x_input: adv_img})

    # Save the adversarial image
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, ...]*255, np.uint8))
      adv_image.save(os.path.join(args.save_dir, '{}_adv.jpg'.format(indices[index])))
    
    # Logging
    if success:
      total_num_corrects += 1
      total_queries.append(num_queries)
      index_to_query[indices[index]] = num_queries
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack succeeds, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        p[0], average_queries, median_queries, success_rate))   
    else:
      index_to_query[indices[index]] = -1
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        p[0], average_queries, median_queries, success_rate))   
    
    index += 1
     

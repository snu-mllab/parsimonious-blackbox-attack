import argparse
import inspect
import json
import numpy as np
import os
from PIL import Image
import sys
import tensorflow as tf

import attacks
from tools.imagenet_labels import *
from tools.inception_v3_imagenet import model
from tools.utils import *

""" The collection of all attack classes """
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" Arguments """
IMAGENET_PATH = '/data_large/readonly/imagenet_data'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--asset_dir', default='./assets', type=str)
parser.add_argument('--save_dir', default='./saves', type=str)

# Experimental Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')

# Attack setting
parser.add_argument('--attack', default='ParsimoniousAttack', type=str, help='The type of attack')
parser.add_argument('--loss_func', default='xent', type=str, help='The type of loss function')
parser.add_argument('--epsilon', default=0.05, type=float, help='The maximum perturbation')
parser.add_argument('--max_queries', default=10000, type=int, help='The query limit')
parser.add_argument('--targeted', action='store_true', help='Targeted attack if true')

# Local Search Setting
parser.add_argument('--max_iters', default=1, type=int, help='The number of iterations in local search')
parser.add_argument('--block_size', default=32, type=int, help='Initial block size')
parser.add_argument('--batch_size', default=64, type=int, help='The size of batch. No batch if negative')
parser.add_argument('--no_hier', action='store_true', help='No hierarchical evaltuation if true')

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
    
  # Create a session
  sess = tf.InteractiveSession()

  # Build a graph
  x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
  y_input = tf.placeholder(dtype=tf.int32, shape=[None])
  
  logits, preds = model(sess, x_input)
  
  model = {
    'x_input': x_input,
    'y_input': y_input,
    'logits': logits,
    'preds': preds,
  }
  
  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(model, args)

  # Create a directory
  if args.save_img:
    tf.gfile.MakeDirs(args.save_dir)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load the indices
  if args.targeted:
    indices = np.load(os.path.join(args.asset_dir, 'indices_targeted.npy'))
  else:
    indices = np.load(os.path.join(args.asset_dir, 'indices_untargeted.npy'))

  # Main loop
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_queries = []
  index_to_query = {}
 
  while count < args.sample_size:
    tf.logging.info('')
    
    # Get an image and the corresponding label
    initial_img, orig_class = get_image(indices[index], IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)

    # Generate a target class (same method as in NES attack)
    if args.targeted:
      target_class = pseudorandom_target(indices[index], NUM_LABELS, orig_class)
      target_class = np.expand_dims(target_class, axis=0)
      
    orig_class = np.expand_dims(orig_class, axis=0)
   
    count += 1
    
    # Run attack
    if args.targeted:
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
        count, indices[index], label_to_name(orig_class[0]), label_to_name(target_class[0])))
      adv_img, num_queries, success = attack.perturb(initial_img, target_class, indices[index], sess)
    else: 
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
        count, indices[index], label_to_name(orig_class[0])))
      adv_img, num_queries, success = attack.perturb(initial_img, orig_class, indices[index], sess)
    
    # Check if the adversarial image satisfies the constraint
    assert np.amax(np.abs(adv_img-initial_img)) <= args.epsilon+1e-3    
    assert np.amax(adv_img) <= 1.+1e-3
    assert np.amin(adv_img) >= 0.-1e-3
    p = sess.run(preds, feed_dict={x_input: adv_img})

    # Save the adversarial image
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :]*255, np.uint8))
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
        label_to_name(p[0]), average_queries, median_queries, success_rate))
    else:
      index_to_query[indices[index]] = -1
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        label_to_name(p[0]), average_queries, median_queries, success_rate))

    index += 1
    

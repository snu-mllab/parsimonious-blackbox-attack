import cv2
import heapq
import math
import numpy as np
import sys
import tensorflow as tf
import time


class LocalSearchHelper(object):
  """A helper for local search algorithm.
  Note that since heapq library only supports min heap, we flip the sign of loss function.
  """

  def __init__(self, model, args, **kwargs):
    """Initalize local search helper.
    
    Args:
      model: TensorFlow model
      loss_func: str, the type of loss function
      epsilon: float, the maximum perturbation of pixel value
    """
    # Hyperparameter setting 
    self.epsilon = args.epsilon
    self.max_iters = args.max_iters
    self.targeted = args.targeted
    self.loss_func = args.loss_func
    
    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']

    probs = tf.nn.softmax(self.logits)
    batch_num = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_num, self.y_input], axis=1)
    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_num, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)

    if self.targeted:
      if self.loss_func == 'xent':
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
      elif self.loss_func == 'cw':
        self.losses = tf.log(max_probs+1e-10) - tf.log(ground_truth_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit() 
    else:
      if self.loss_func == 'xent':
        self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
      elif self.loss_func == 'cw':
        self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit() 
 
  def _perturb_image(self, image, noise):
    """Given an image and a noise, generate a perturbed image.
    First, resize the noise with the size of the image. 
    Then, add the resized noise to the image. 
    
    Args:
      image: numpy array of size [1, 299, 299, 3], an original image
      noise: numpy array of size [1, 256, 256, 3], a noise

    Returns:
      adv_iamge: numpy array with size [1, 299, 299, 3], a perturbed image
    """ 
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image 

  def _flip_noise(self, noise, block):
    """Flip the sign of perturbation on a block.
    Args:
      noise: numpy array of size [1, 256, 256, 3], a noise
      block: [upper_left, lower_right, channel], a block
    
    Returns:
      noise: numpy array of size [1, 256, 256, 3], an updated noise 
    """
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  def perturb(self, image, noise, label, sess, blocks):		
    """Update a noise with local search algorithm.
    
    Args:
      image: numpy array of size [1, 299, 299, 3], an original image
      noise: numpy array of size [1, 256, 256, 3], a noise
      label: numpy array of size [1], the label of image (or target label)
      sess: TensorFlow session
      blocks: list, a set of blocks

    Returns: 
      noise: numpy array of size [1, 256, 256, 3], an updated noise
      num_queries: int, the number of queries
      curr_loss: float, the value of loss function
      success: bool, True if attack is successful
    """
    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]
   
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Check if a block is in the working set or not
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If the sign of perturbation on the block is positive,
      # which means the block is in the working set, then set A to 1
      if noise[0, x, y, channel] > 0:
        A[i] = 1

    # Calculate the current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]
    
    # Early stopping
    if self.targeted:
      if preds == label:
        return noise, num_queries, curr_loss, True
    else:
      if preds != label:
        return noise, num_queries, curr_loss, True
  
    # Main loop
    for _ in range(self.max_iters):
      # Lazy greedy insert
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1
          return noise, num_queries, curr_loss, True 
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))
      
      # Pick the best element and insert it into the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 1
      
      # Add elements into the working set
      while len(priority_queue) > 0:
        # Pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)

        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1
        margin = losses[0]-curr_loss
        
        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Update the noise
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 1
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
	    
      priority_queue = []

      # Lazy greedy delete
      indices,  = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1
          return noise, num_queries, curr_loss, True 
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))

      # Pick the best element and remove it from the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 0
      
      # Delete elements into the working set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1 
        margin = losses[0]-curr_loss
      
        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Update the noise
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 0
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss, False


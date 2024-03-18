import tensorflow as tf
import numpy as np

# feature map size 키울때 
resized_image = tf.image.resize(img, [size[0], size[1]])

# concatenate
concatenated_tensor = tf.concat([tensor1, tensor2], axis=3)

# session 잡아줄때
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(cc))
sess.close()

# pdb for loop
import code; code.interact(local=vars())
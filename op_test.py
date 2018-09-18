import tensorflow as tf
import numpy as np
my_module = tf.load_op_library('./Mysqure.so')
input_tensor = tf.Variable(np.array([1,5,36,3.25]),dtype = np.float64)
output_tensor = my_module.mysqure(input_tensor)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(output_tensor))
	
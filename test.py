import tensorflow as tf
my_module = tf.load_op_library('./Mysqure.so')

with tf.Session():
	print(my_module.mysqure([[1, 2], [3, 4]]).eval())
	
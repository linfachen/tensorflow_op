import tensorflow as tf
from tensorflow.python.framework import ops

my_module = tf.load_op_library('./Mysqure.so')

mysqure = my_module.mysqure

@ops.RegisterGradient("Mysqure")
def _register_mysqure(op,grad):
	return my_module.mysqure_grad(grad,op.inputs[0])
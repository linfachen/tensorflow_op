import tensorflow as tf 
import numpy as np 
import unittest
import os
import mysqure
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

class Test(unittest.TestCase):  

    def test_b_run(self):

        input_tensor = tf.constant(np.random.rand(250,256)*10,dtype = np.float32)
        output_tensor1 = mysqure.mysqure(input_tensor)
        output_tensor2 = input_tensor * input_tensor

        grad1 = tf.gradients(output_tensor1,input_tensor)
        grad2 = tf.gradients(output_tensor2,input_tensor)

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            res1,res2 = sess.run([grad1,grad2])

        print("="*40,"res1","="*40)	
        print(res1)
        print("="*40,"res2","="*40)	
        print(res1)	
        np.testing.assert_equal(res1,res2)  

if __name__ == '__main__':
    unittest.main()

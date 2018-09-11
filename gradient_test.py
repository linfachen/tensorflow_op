import tensorflow as tf 
import numpy as np 
import unittest

import mysqure


class Test(unittest.TestCase):  

    def test_b_run(self):
        input_tensor = tf.constant(np.array([1,5,36,3.25]),dtype = np.float32)
        output_tensor1 = mysqure.mysqure(input_tensor)
        output_tensor2 = input_tensor * input_tensor

        grad1 = tf.gradients(output_tensor1,input_tensor)
        grad2 = tf.gradients(output_tensor2,input_tensor)

        with tf.Session() as sess:
            res1,res2 = sess.run([grad1,grad2])

        print("="*40,"res1","="*40)	
        print(res1)
        print("="*40,"res2","="*40)	
        print(res1)	
        np.testing.assert_equal(res1,res2)  

if __name__ == '__main__':
    unittest.main()

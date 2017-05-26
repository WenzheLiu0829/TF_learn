import tensorflow as tf
import numpy as np
##Save to file
'''
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'my_net/save_net.ckpt')

    print("Save to path:", save_path)'''


#restore variables
#redefine the same shape and same type for your variables

'''
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32, name='biases')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net.ckpt')
    print("Weights:", sess.run(W))
    print("biases:", sess.run(b))
    '''
npc = np.arange(100)
print(npc)
tfc = tf.Variable(npc) # Use variable 


with tf.Session() as sess:   
    tf.initialize_all_variables().run() # need to initialize all variables

    print('tfc:\n', tfc.eval())
    print('npc:\n', npc)
    for i in range(100):
        npc[i] = 0
    tfc.assign(npc).eval() # assign_sub/assign_add is also available.
    print('modified tfc:\n', tfc.eval())
    print('modified npc:\n', npc)

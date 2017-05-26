import tensorflow as tf

#About placeholder

input1 = tf.placeholder(tf.float32) #type usually is 32
#input1 = tf.placeholder(tf.float32,[2,2]) #2*2 input
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)

with tf.Session() as sess:
	print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
	#when using placeholder, must use run.feed_dict
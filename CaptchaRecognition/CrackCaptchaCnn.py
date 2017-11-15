# coding=utf-8

from __future__ import print_function
import tensorflow as tf
from load_data import *
import random

class CrackCaptchaCnn(object):

	learning_rate = 0.001
	training_iters = 20000
	batch_size = 128
	display_step = 10

	n_input = 784 # data input (img shape: 28*28)
	n_classes = 8 # total classes (2-9 digits)
	dropout = 0.75 # Dropout, probability to keep units

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

	# Store layers weight & bias
	weights = {
		# 5x5 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		# 5x5 conv, 32 inputs, 64 outputs
		'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		# fully connected, 7*7*64 inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))
	}

	def __init__(self):
		super(CrackCaptchaCnn,self).__init__()

	def conv2d(self,x,W,b,strides = 1):
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)

	def maxpool2d(self,x, k=2):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

	def conv_net(self,x,weights,biases,dropout):
		# Reshape input picture
		x = tf.reshape(x, shape=[-1, 28, 28, 1])
		# Convolution Layer
		conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
		# Max Pooling (down-sampling)
		conv1 = self.maxpool2d(conv1, k=2)
		# Convolution Layer
		conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
		# Max Pooling (down-sampling)
		conv2 = self.maxpool2d(conv2, k=2)
		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, dropout)
		# Output, class prediction
		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		return out

	def train_crack_captcha_cnn(self):
		pred = self.conv_net(self.x,self.weights,self.biases,self.dropout)

		# Define loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
		# Evaluate model
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		# Initializing the variables
		init = tf.global_variables_initializer()
		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			step = 1
			train_dataset, train_labels,test_dataset, test_labels, label_map=load_model()
			while step * self.batch_size < self.training_iters:
				offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
				batch_data = train_dataset[offset:(offset + self.batch_size), :]
				batch_labels = train_labels[offset:(offset + self.batch_size), :]
				sess.run(optimizer, feed_dict={self.x: batch_data, self.y: batch_labels,
                                       self.keep_prob: self.dropout})
				if step % self.display_step == 0:
					loss, acc = sess.run([cost, accuracy], feed_dict={self.x: batch_data,
                                                              self.y: batch_labels,
                                                              self.keep_prob: 1.})
					print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
				step += 1
			print("Optimization Finished!")
			print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={self.x: test_dataset,
                                      self.y: test_labels,
                                      self.keep_prob: 1.}))

	def get_name_and_image(self):
		base_dir = "E:/pest1/training_set/2/"
		all_image = os.listdir(base_dir)
		random_file = random.randint(0,2)
		base = os.path.basename('E:/pest1/training_set/2/'+ all_image[random_file])
		name = os.path.splitext(base)[0]
		image = Image.open('E:/pest1/training_set/2/'+ all_image[random_file])
		image = np.array(image)
		return name,image

	def crack_captcha_test(self):
		pred = self.conv_net(self.x,self.weights,self.biases,self.dropout)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			# saver.restore(sess,tf.train.latest_checkpoint('.'))
			n = 1
			while n < 10:
				text,image = self.get_name_and_image()
				print ('text:',text)
				image = 1 * (image.flatten())
				# predict = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
				predict = tf.argmax(tf.reshape(pred, [-1,1,1]),2)
				print ('predict:',predict)
				text_list = sess.run(predict,feed_dict = {self.x:[image],self.keep_prob : 1})
				print ("text_list:",text_list)

if __name__ == '__main__':
	job = CrackCaptchaCnn()
	job.crack_captcha_test()
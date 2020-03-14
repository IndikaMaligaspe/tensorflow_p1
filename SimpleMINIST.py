import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#use TF helper function to pull down MNIST data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#x is the placeholder for 28 *28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

#y_ is a 10 element vector containing the predicated digit (0-9) "y bar" - 
#[0,1,0,0,0,0,0,0,0,0] for 1, [0,0,0,0,0,0,0,0,0,1] for 9
y_ = tf.placeholder(tf.float32, [None, 10])

#define weights and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#define the inference model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

#each training step in gradient decet we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize the global variables
init = tf.global_variables_initializer()

#create an interactive session that can span multiple code blocks.
sess = tf.Session()

#perform the global variable initializer
sess.run(init)

#perform 1000 training steps
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #get 100 random data points from data.  
                                                     #batch_xs= image , batch_ys = digit(0-9)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

#evaluate how well the model did. Do this by comparing the digit with the highest probability in actula(y) and predcited (y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_acciracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("Test Accuracy: {0}%".format(test_acciracy*100.00))
sess.close()


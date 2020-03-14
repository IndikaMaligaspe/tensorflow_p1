'''
  This is a simple prediction of house prices based on house size.
'''
import matplotlib
#matplotlib.use('tkagg')

import tensorflow as tf
import numpy as np
import math
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#generate some houses with sizes between 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
#plot generated house size

#Generate houser prices from house size with reandom noise added
np.random.seed(42)
house_price = house_size *100.0 + np.random.randint(low=20000, high=70000, size=num_house)



'''
plt.plot(house_size, house_price, "bx")  #bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")


plt.show()
'''

def normalize(array):
  return (array - array.mean()) / array.std()

num_train_samples = math.floor(num_house * 0.7)

train_price = np.asanyarray(house_price[:num_train_samples:])

#We initialize them to some random values based on the normal distribution
num_train_samples = math.floor(num_house * 0.7)
#define training data

train_price_norm = normalize(train_price)
train_house_size = np.asarray(house_size[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
#define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])


test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")
#Set up the TF placeholders that get updated as we decend down the gradient

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float",name="price")

#Define the variables holding the size_factor and price we set during training , we initilize them with some random value
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")

#Define the operations for the predicting values - predicted price = (size_factor * hour_size) + price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

#Define the loss function (how much error) - Mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

#Optimze the learning rate. How many steps down the gradient
learning_rate = 0.1

#Define the gradient descent optimizer that will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#Initialize the variables
init = tf.global_variables_initializer()

#Launch the graph in the session
with tf.Session() as sess:
  sess.run(init)

  #set how often to display traiing progress and number of training iterations
  display_every = 2
  num_training_iter = 50

  #calculate the number of lines to animation
  fit_num_plots = math.floor(num_training_iter/display_every)
  #add storage of factor and offset values from each epoch
  fit_size_factor = np.zeros(fit_num_plots)
  fit_price_offset = np.zeros(fit_num_plots)
  fit_plot_idx = 0

  #keep iterating the training data
  for iteration in range(num_training_iter):
    # fit all training data
    for(x,y) in zip(train_house_size_norm, train_price_norm):
      sess.run(optimizer, feed_dict={tf_house_size:x, tf_price:y})

    #display current status
    if (iteration + 1) % display_every == 0:
      c = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm,tf_price:train_house_size_norm})
      print("iteration #:", '%04d' % (iteration +1), "cost=","{:.9f}".format(c), \
      "size_factor=", sess.run(tf_size_factor), "pice_offset=", sess.run(tf_price_offset))

      #Save the fit size_factor and price_offset to allow animation of learning process
      fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
      fit_price_offset[fit_plot_idx] = sess.run(tf_price_offset)
      fit_plot_idx = fit_plot_idx + 1
  print("Optimization Finished!")
  training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})
  print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

  train_house_size_mean = train_house_size.mean()
  train_house_size_std = train_house_size.std()

  train_price_mean = train_price.mean()
  train_price_std = train_price.std()


  fig,ax = plt.subplots()
  line = ax.plot(house_size,house_price)

  plt.rcParams["figure.figsize"] = (10,8)
  plt.title("Gradient Descent Fitting Regression Line")
  plt.ylabel("Price")
  plt.xlabel("Size (sq.ft)")
  plt.plot(train_house_size, train_price, 'go', label='Training data')
  plt.plot(test_house_size,test_house_price,'mo', label='Testing data')

  def animate(i):
    line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean) #update data
    line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offset[i]) * train_price_std + train_price_mean)
    return line

  #init only required for blitting to give a clean slate
  def initAnim():
    line.set_ydata(np.zeros(shape=house_price.shape[0])) #set y's to 0
    return line
  
  ani = animation.FuncAnimation(fig,animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim, interval=1000, blit=True)

  plt.show()
  

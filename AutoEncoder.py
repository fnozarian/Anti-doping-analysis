import tensorflow as tf
import numpy as np
from pandas import DataFrame
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# preparing dataset
data = DataFrame.from_csv("testosterone_levels.csv")
testosterones = [person['Testosterone'].values for _, person in data.groupby("Person")]
padded_testosterones = np.array(pad_sequences(testosterones, dtype='float32'))

# setting parameters
learning_rate = 0.01
epochs = 1000
batch_size = 32

input_dim = padded_testosterones.shape[1]  # number of features or longest testosterone seq
num_hidden_1 = 256
num_hidden_2 = 128
num_hidden_3 = 64

test_step = 5


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, input_dim])


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape, init_val=0.1):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(init_val, shape=shape)

    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer."""

    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

    return activations


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def encoder(x):
    with tf.name_scope("encoder"):
        encoder_h1 = nn_layer(x, input_dim, num_hidden_1, "encoder_h1")
        encoder_h2 = nn_layer(encoder_h1, num_hidden_1, num_hidden_2, "encoder_h2")
        encoder_h3 = nn_layer(encoder_h2, num_hidden_2, num_hidden_3, "encoder_h3")

    return encoder_h3

def decoder(x):
    with tf.name_scope("decoder"):
        decoder_h1 = nn_layer(x, num_hidden_3, num_hidden_2, "decoder_h1")
        decoder_h2 = nn_layer(decoder_h1, num_hidden_2, num_hidden_1, "decoder_h2")
        decoder_h3 = nn_layer(decoder_h2, num_hidden_1, input_dim, "decoder_h3", act=tf.identity) # do not apply activation of output layer!

    return decoder_h3


encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = x


with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.square(y_true - y_pred))

tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/logs/train', sess.graph)
test_writer = tf.summary.FileWriter('/logs/test')

x_train, x_test = train_test_split(padded_testosterones, test_size=.33)
train_indices = np.arange(len(x_train))



# Train the model, and also write summaries.
# Every `test_step`th step, measure test-set loss, and write test summaries
# All other steps, run `train_step` on training data, and write training summaries
for i in range(epochs):

    np.random.shuffle(train_indices)

    for b in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[b: b + batch_size]
        x_batch = x_train[batch_indices, :]
        summary, _ = sess.run([merged, train_step], feed_dict={x: x_batch})
        train_writer.add_summary(summary, i)

    if i % test_step == 0:
        summary = sess.run(merged, feed_dict=x_test)
        test_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()


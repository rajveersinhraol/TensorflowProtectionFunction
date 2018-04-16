import numpy as np
import tensorflow as tf

# Parameters
#-----------------------------------------------------------------------/
logs_path = '/tmp/tensorflow_logs/example'

#-----------------------------------------------------------------------/
DATA_LENGTH_SECONDS = 10
SYSTEM_FREQUENCY = 60
PACKETS_PER_CYCLE = 8
NumCycles = np.ceil(DATA_LENGTH_SECONDS/SYSTEM_FREQUENCY)
FRAME_LENGTH = PACKETS_PER_CYCLE * NumCycles

#-----------------------------------------------------------------------/
V = tf.placeholder(tf.float32, [FRAME_LENGTH, 3], name='InputVoltage')
I = tf.placeholder(tf.float32, [FRAME_LENGTH, 3], name='InputCurrent')
y = V*2

#-----------------------------------------------------------------------/
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    rand_array =np.array(np.random.rand(int(FRAME_LENGTH), 3))
    print(sess.run(y, feed_dict={V: rand_array}))

#-----------------------------------------------------------------------/
print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
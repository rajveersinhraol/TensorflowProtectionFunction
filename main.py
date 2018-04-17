import numpy as np
import tensorflow as tf


# Parameters
#-----------------------------------------------------------------------/
logs_path = '/tmp/tensorflow_logs/example'

#-----------------------------------------------------------------------/
DATA_LENGTH_SECONDS = 10
SYSTEM_FREQUENCY = 60
PACKETS_PER_CYCLE = 8
NumCycles = int(np.ceil(DATA_LENGTH_SECONDS*SYSTEM_FREQUENCY))
TOTAL_PACKETS = int(PACKETS_PER_CYCLE * NumCycles)
print("Num of cycles = " + str(NumCycles))
print("total number of packets  = " + str(TOTAL_PACKETS))
print("Packets per cycle  = " + str(PACKETS_PER_CYCLE))

        
def Dft(A,B,C):
    #DFT constanst
    FUND_FACTOR=0.18138511
    # A = tf.slice(input, [0, 0], [PACKETS_PER_CYCLE, 1])
    # B = tf.slice(input, [0, 1], [PACKETS_PER_CYCLE, 2])
    # C = tf.slice(input, [0, 2], [PACKETS_PER_CYCLE, 3])
    with tf.name_scope('Dft'):
        real = (A[0] - A[4] + (1.0 / np.sqrt(2)) * (A[1] - A[3] - A[5] + A[7])) * FUND_FACTOR
        imag = ( - A[2] + A[6] + (1.0 / np.sqrt(2)) * ( - A[1] - A[3] + A[5] + A[7])) * FUND_FACTOR
        FundMagA = tf.sqrt(real * real + imag * imag)

        real = (B[0] - B[4] + (1.0 / np.sqrt(2)) * (B[1] - B[3] - B[5] + B[7])) * FUND_FACTOR
        imag = ( - B[2] + B[6] + (1.0 / np.sqrt(2)) * ( - B[1] - B[3] + B[5] + B[7])) * FUND_FACTOR
        FundMagB = tf.sqrt(real * real + imag * imag)

        real = (C[0] - C[4] + (1.0 / np.sqrt(2)) * (C[1] - C[3] - C[5] + C[7])) * FUND_FACTOR
        imag = ( - C[2] + C[6] + (1.0 / np.sqrt(2)) * ( - C[1] - C[3] + C[5] + C[7])) * FUND_FACTOR
        FundMagC = tf.sqrt(real * real + imag * imag)
    return FundMagA, FundMagB, FundMagC
def main():
    #-----------------------------------------------------------------------/
    Va = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageA')
    Vb = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageB')
    Vc = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageC')
    I = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE, 3], name='InputCurrent')
    y = tf.placeholder(tf.float32, [None, 3], name='output')
    y = Dft(Va,Vb,Vc)

    #-----------------------------------------------------------------------/
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sin_wave = np.zeros((TOTAL_PACKETS, 3))
        A=69
        sin_wave[:,0] = np.linspace( 0, 2*np.pi, TOTAL_PACKETS)
        sin_wave[:,1] = np.linspace( - (np.pi + np.pi/3), (np.pi + np.pi/3), TOTAL_PACKETS)
        sin_wave[:,2] = np.linspace( + (np.pi + np.pi / 3), - (np.pi + np.pi / 3), TOTAL_PACKETS)
        for i in range(NumCycles):
            magA,magb,magc=sess.run([y[0],y[1],y[2]], feed_dict={Va: A*sin_wave[i*PACKETS_PER_CYCLE:(i+1)*PACKETS_PER_CYCLE,0],Vb: A*sin_wave[i*PACKETS_PER_CYCLE:(i+1)*PACKETS_PER_CYCLE,1],Vc: A*sin_wave[i*PACKETS_PER_CYCLE:(i+1)*PACKETS_PER_CYCLE,2]})
    #-----------------------------------------------------------------------/
    print("Run the command line:\n" \
            "--> tensorboard --logdir=/tmp/tensorflow_logs " \
            "\nThen open http://0.0.0.0:6006/ into your web browser")

    #-----------------------------------------------------------------------/
main()

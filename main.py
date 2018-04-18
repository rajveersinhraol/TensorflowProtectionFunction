import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Parameters
#-----------------------------------------------------------------------/
logs_path = '/tmp/tensorflow_logs/example'

#-----------------------------------------------------------------------/
DATA_LENGTH_SECONDS = 10
SYSTEM_FREQUENCY = 60
PACKETS_PER_CYCLE = 8

NOMINAL = 1
NumCycles = int(np.ceil(DATA_LENGTH_SECONDS*SYSTEM_FREQUENCY))
TOTAL_PACKETS = int(PACKETS_PER_CYCLE * NumCycles)
print("Num of cycles = " + str(NumCycles))
print("total number of packets  = " + str(TOTAL_PACKETS))
print("Packets per cycle  = " + str(PACKETS_PER_CYCLE))

t = np.linspace(0.0, DATA_LENGTH_SECONDS, TOTAL_PACKETS)
sin_wave = np.zeros((TOTAL_PACKETS, 3))
sin_wave[:, 0] = np.sin(2 * np.pi * 60 * t)
sin_wave[:, 1] = np.sin(2 * np.pi * 60 * t + np.pi)
sin_wave[:, 2] = np.sin(2 * np.pi * 60 * t - np.pi)

def Seq(A, B, C):
    with tf.name_scope('Seq'):
        Preal = [A[0], B[0], C[0]]
        Pimag = [A[1], B[1], C[1]]       
        BRealPlusCReal = Preal[1] + Preal[2]
        BRealMinusCReal = (Preal[1] - Preal[2])
        
        BImagPlusCImag = Pimag[1] + Pimag[2]
        BImagMinusCImag = Pimag[1] - Pimag[2]
        
        Pos_real_part = (Preal[0] - BRealPlusCReal * 0.5 - BImagMinusCImag * (np.sqrt(3) / 2.0)) * (1.0 / 3.0)
        Pos_imag_part = (Pimag[0] + BRealMinusCReal * (np.sqrt(3) / 2.0) - BImagPlusCImag * 0.5) * (1.0 / 3.0)

        Neg_real_part = (Preal[0] - BRealPlusCReal * 0.5 + BImagMinusCImag * (np.sqrt(3) / 2.0)) * (1.0 / 3.0)
        Neg_imag_part = (Pimag[0] - BRealMinusCReal * (np.sqrt(3) / 2.0) - BImagPlusCImag * 0.5) * (1.0 / 3.0)

        Zero_real_part = (Preal[0] + Preal[1] + Preal[2]) / 3
        Zero_imag_part = (Pimag[0] + Pimag[1] + Pimag[2]) / 3
    return [Pos_real_part,Pos_imag_part],[Neg_real_part,Neg_imag_part],[Zero_real_part,Zero_imag_part]
        
def Dft(A,B,C):
    #DFT constanst
    FUND_FACTOR=0.18138511
    # A = tf.slice(input, [0, 0], [PACKETS_PER_CYCLE, 1])
    # B = tf.slice(input, [0, 1], [PACKETS_PER_CYCLE, 2])
    # C = tf.slice(input, [0, 2], [PACKETS_PER_CYCLE, 3])
    with tf.name_scope('Dft'):
        realA = (A[0] - A[4] + (1.0 / np.sqrt(2)) * (A[1] - A[3] - A[5] + A[7])) * FUND_FACTOR
        imagA= ( - A[2] + A[6] + (1.0 / np.sqrt(2)) * ( - A[1] - A[3] + A[5] + A[7])) * FUND_FACTOR

        realB = (B[0] - B[4] + (1.0 / np.sqrt(2)) * (B[1] - B[3] - B[5] + B[7])) * FUND_FACTOR
        imagB= ( - B[2] + B[6] + (1.0 / np.sqrt(2)) * ( - B[1] - B[3] + B[5] + B[7])) * FUND_FACTOR

        realC = (C[0] - C[4] + (1.0 / np.sqrt(2)) * (C[1] - C[3] - C[5] + C[7])) * FUND_FACTOR
        imagC = ( - C[2] + C[6] + (1.0 / np.sqrt(2)) * ( - C[1] - C[3] + C[5] + C[7])) * FUND_FACTOR

    return [realA,imagA],[realB,imagB],[realC,imagC]

def MagAngle(A, B, C):
    with tf.name_scope('Mag_Ang_Calc'):
        real = A[0]
        imag = A[1] 
        FundMagA = tf.sqrt(real * real + imag * imag)
        AngA = tf.atan(A[1]/A[0])
        real = B[0]
        imag = B[1] 
        FundMagB = tf.sqrt(real * real + imag * imag)
        AngB = tf.atan(B[1]/B[0])
        real = C[0]
        imag = C[1] 
        FundMagC = tf.sqrt(real * real + imag * imag)
        AngC = tf.atan(C[1]/C[0])
    return [FundMagA,AngA],[FundMagB,AngB],[FundMagC,AngC]
    
def WaveGen(i,Amplitude):
    with tf.name_scope('WaveGenerator'):
        a = Amplitude * sin_wave[i * PACKETS_PER_CYCLE:(i + 1) * PACKETS_PER_CYCLE, 0]
        b = Amplitude * sin_wave[i * PACKETS_PER_CYCLE:(i + 1) * PACKETS_PER_CYCLE, 1]
        c = Amplitude * sin_wave[i * PACKETS_PER_CYCLE:(i + 1) * PACKETS_PER_CYCLE, 2]*0.0
    return a, b, c

def CheckIndividualTrip(FundMagA,FundMagB,FundMagC):
    TripA = tf.less( FundMagA , 0.04 * NOMINAL)
    TripB = tf.less( FundMagB , 0.04 * NOMINAL)
    TripC = tf.less(FundMagC, 0.04 * NOMINAL)
    return TripA, TripB, TripC

def IndividualTripFalse(FundMagA,FundMagB,FundMagC):
    TripA = tf.constant(False)
    TripB = tf.constant(False)
    TripC = tf.constant(False)
    return TripA,TripB,TripC
def dev46BC(PosSeqMagAng, NegSeqMagAng, MagAngA, MagAngB, MagAngC):
    with tf.name_scope('46BC'):
        PosSeqMag = PosSeqMagAng[0]
        NegSeqMag = NegSeqMagAng[0]
        FundMagA = MagAngA[0]
        FundMagB = MagAngB[0]
        FundMagC = MagAngC[0]
        Enable = tf.constant(1, name='Enable')
        PickupLevel = tf.constant(50.0 / 100, name='PickupLevel')
        Trip = tf.logical_and(tf.greater(NegSeqMag / PosSeqMag, PickupLevel), tf.greater(3 * PosSeqMag, 0.04 * NOMINAL),name='logic')
        TripA, TripB, TripC=tf.cond(tf.equal(Trip, tf.constant(True)), lambda: CheckIndividualTrip(FundMagA,FundMagB,FundMagC), lambda: IndividualTripFalse(FundMagA,FundMagB,FundMagC),name='TripCheck')
    return TripA, TripB, TripC

def main():
    #-----------------------------------------------------------------------/
    Va = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageA')
    Vb = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageB')
    Vc = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputVoltageC')
    Ia = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputCurrentA')
    Ib = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputCurrentB')
    Ic = tf.placeholder(tf.float32, [PACKETS_PER_CYCLE], name='InputCurrentC')
    
    complexVs = Dft(Va, Vb, Vc)
    seqV = Seq(complexVs[0], complexVs[1], complexVs[2])
    seqVma = MagAngle(seqV[0], seqV[1], seqV[2])
    funMA = MagAngle(complexVs[0], complexVs[1], complexVs[2])
    
    complexIs = Dft(Ia, Ib, Ic)
    seqI = Seq(complexIs[0],complexIs[1],complexIs[2])
    seqIMA = MagAngle(seqI[0], seqI[1], seqI[2])
    funIMA=MagAngle(complexIs[0],complexIs[1],complexIs[2])
    output=dev46BC(seqIMA[0],seqIMA[1],funIMA[0],funIMA[1],funIMA[2])
    #-----------------------------------------------------------------------/
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        for i in range(NumCycles):
            a,b,c=WaveGen(i,NOMINAL)
            sess.run([output[0], output[1], output[2]], feed_dict={Ia: a, Ib: b, Ic: c})
    
    #-----------------------------------------------------------------------/
    print("Run the command line:\n" \
            "--> tensorboard --logdir=/tmp/tensorflow_logs " \
            "\nThen open http://0.0.0.0:6006/ into your web browser")

    #-----------------------------------------------------------------------/
main()

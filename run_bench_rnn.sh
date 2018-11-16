##############################################
### wrapper script for running rnn benchmarks
##############################################


BENCH_RNN_TRAIN=./bench-rnn-train
[ -e $BENCH_RNN_TRAIN ] || cp ./build/examples/bench-rnn-train $BENCH_RNN_TRAIN

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"

export MKLDNN_VERBOSE=1

#for i in `seq 1 10` ;
#do
./$BENCH_RNN_TRAIN 64 50 500 500
./$BENCH_RNN_TRAIN 128 50 1024 1024
./$BENCH_RNN_TRAIN 128 50 4096 4096
#done

#'''
#./$BENCH_RNN_TRAIN 64 50 400 500
#./$BENCH_RNN_TRAIN 64 50 400 500
#./$BENCH_RNN_TRAIN 64 50 10 20
#./$BENCH_RNN_TRAIN 64 50 10 20
#./$BENCH_RNN_TRAIN 64 50 400 500
#./$BENCH_RNN_TRAIN 64 50 400 500
#./$BENCH_RNN_TRAIN 64 50 400 500
#'''

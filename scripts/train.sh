DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/tmp

#python run.py \
#    --DATA_LIST=$OUTPUT/aistudio/data/ \
#    --BATCH_SIZE=64 \
#    --EPOCH_NUM=2 \
#    --MODEL=$OUTPUT/aistudio/work/infer_model/ \
#    --IS_DRAW

python run.py \
    --DATA_LIST=$OUTPUT/aistudio/data/ \
    --BATCH_SIZE=64 \
    --EPOCH_NUM=10 \
    --MODEL=$OUTPUT/aistudio/work/infer_model/
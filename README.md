To pretrain on step 1:
```
CUDA_VISIBLE_DEVICES=0 retinanet/tools/train_adapt.sh \
  --logging=20 --model-steps=1 --epochs=1 --batches-per-epoch=5000 \
  --tensorboard-dir=logs/Aug17/step1 --tensorboard-freq=10 \
  --snapshot-dir=snapshots/Aug17/step1
```



To continue training.

To try something out:
```
CUDA_VISIBLE_DEVICES=0 retinanet/tools/train_adapt.sh \
  --logging=10 --model-steps=2,3 --epochs=1 --batches-per-epoch=50 \
  --load-snapshot snapshots/Aug16/step1/epoch000.h5
```

To actually train:
```
CUDA_VISIBLE_DEVICES=0 retinanet/tools/train_adapt.sh \
  --logging=20 --model-steps=2,3 --epochs=1 --batches-per-epoch=5000 \
  --tensorboard-dir="logs/Aug16/step23_cont" --tensorboard-freq=10 --tensorboard-offset=500 \
  --load-snapshot snapshots/Aug16/step1/epoch000.h5 --snapshot-dir="snapshots/Aug16/step23_init1at500" 
```

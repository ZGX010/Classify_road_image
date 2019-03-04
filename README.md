# Classify_road_image
<img src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif" width=425 height=240 />
<br>
<div align=center><img width="520" height="220" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

```Python
CUDA_VISIBLE_DEVICES=1 \
python train_image_classifier.py \
--train_dir='./tmp/data/mydata/train_logs' \
--dataset_name=mydata \
--dataset_split_name=train \
--dataset_dir=./tmp/data/mydata
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
--train_image_size=1400
```
<br>

```Python
CUDA_VISIBLE_DEVICES=1 \
python eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=./tmp/data/mydata/train_logs \
--dataset_dir=./tmp/data/mydata \
--dataset_name=mydata \
--dataset_split_name=validation \
--model_name=inception_v4
```
<br>

```Python
CUDA_VISIBLE_DEVICES=1 \
python predict.py \
--model_name=inception_v4 \
--predict_file='./103842795-000468-000468.JPG' \
--checkpoint_path=./tmp/data/mydata/train_logs \
--dataset_dir=./tmp/data/mydata \
--predict_image_size=4096
```
<br>

```Python
CUDA_VISIBLE_DEVICES=1 \
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

```Python
CUDA_VISIBLE_DEVICES=1 \
python freeze_graph.py \
--input_graph=./tmp/data/mydata/inception_v4_inf_graph.pb \
--input_checkpoint=./tmp/data/mydata/train_logs/model.ckpt-51498 \
--input_binary=true \
--output_node_names=InceptionV4/Logits/Predictions \
--output_graph=./tmp/data/mydata/crake_classify.pb
```
<br>

```Python
CUDA_VISIBLE_DEVICES=0 \
python classify_image_inception_v3.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

# Classify_road_image
## 项目实现的功能：<br>
本项目实现了基于InceptionV4网络实现的路面影像分类<br>
当运行推理脚本时，tensorflow会从指定的文件夹读取图像输入神经网络并计算每种类别的得分，将得分最高的种类判定为图像所属种类．并将图片移动至对应种类的文件夹<br>
<div align=center><img width="520" height="320" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

## ＩnceptionV4版本的图像分类网络

<img src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png" width=1000 height=629 />
<img src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png" width=1000 height=337 />
<br>

## 运行环境
* tensorflow 1.6
* opencv3 for python2.7
* ubuntu 16.04
* CUDA8.0 & CUdnn6.0

## 数据准备

## 训练模型
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

## 评价模型
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

## 导出模型结构
```Python
CUDA_VISIBLE_DEVICES=1 \
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

## 为模型架构载入训练好的参数
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

## 加载模型并进行推理
```Python
CUDA_VISIBLE_DEVICES=0 \
python classify_image_inception_v3.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

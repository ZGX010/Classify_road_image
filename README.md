# Classify_road_image
## 1.项目实现功能：<br>
>本项目在路面影像上训练Inception_V4模型，实现对的路面影像按有无病害进行分类，测试集上准确率为：，同时提供一个已经在病害影像上训练好的持久化模型，方便进行进一步训练，以适用于自己的数据，并缩短训练时间．<br>
>当运行［8］推理脚本时，脚本将载入训练好的ＰＢ模型，Ｔensorflow会从指定的文件夹读取图像，并输入神经网络计算两种类别分别的得分．依据得分将图片移动至对应的文件夹<br>
<div align=center><img width="520" height="320" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

## 2.Ｉnception_V4 图像分类网络结构

<div align=center><img width="1000" height="629" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png"/></div>
<div align=center><img width="1000" height="337" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png"/></div>
<br>

## 3.运行环境
* tensorflow 1.6
* opencv3 for python2.7
* ubuntu 16.04
* CUDA8.0 & CUdnn6.0

## 4.数据准备
### 4.1克隆代码至本地：<br>
```Python
git clone https://github.com/ZGX010/Classify_road_image.git
```
### 4.2运行环境检测
>在文件目录下运行检测
```Ｐython
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```
### 4.3将需要训练的图像数据处理为ＴＦＲ格式
>将需要训练的数据转为ＴＦＲecord格式，因为在脚本中已经添加了mydata数据集，所以直接将需要训练的数据集，命名为mydata即可．
```python
download_and_convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```

## 5.训练模型
```Python
python train_image_classifier.py \
--train_dir='./tmp/data/mydata/train_logs' \
--dataset_name=mydata \
--dataset_split_name=train \
--dataset_dir=./tmp/data/mydata
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits 
```
<br>

## 6.评价模型
```Python
python eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=./tmp/data/mydata/train_logs \
--dataset_dir=./tmp/data/mydata \
--dataset_name=mydata \
--dataset_split_name=validation \
--model_name=inception_v4
```
<br>

## 7.导出模型
### 7.1导出结构
```Python
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

### 7.2为模型架构载入训练好的参数
```Python
python freeze_graph.py \
--input_graph=./tmp/data/mydata/inception_v4_inf_graph.pb \
--input_checkpoint=./tmp/data/mydata/train_logs/model.ckpt-51498 \
--input_binary=true \
--output_node_names=InceptionV4/Logits/Predictions \
--output_graph=./tmp/data/mydata/crake_classify.pb
```
<br>

## 8.加载模型并进行推理
```Python
python classify_image_inception_v3.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

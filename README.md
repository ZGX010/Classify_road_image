# Classify_road_image
## 1  Project Function：<br>
This project trains the Inception_V4 model on the road image to realize the classification of the road image according to the disease. The accuracy of the test set is:0.95． At the same time, I provide a model that has been trained on the disease image to facilitate further training. Apply to your own data and shorten training time.<br>
When you run the [8] inference script, the trained PB model will be loaded, and Tensorflow will read the image from the specified folder and enter the neural network to calculate the scores for the two categories. Move the image to the corresponding folder based on the score<br>
The following GIF is the script running process．<br>
<div align=center><img width="450" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

## 2  Inception_V4 model

<div align=center><img width="800" height="480" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png"/></div>
<div align=center><img width="800" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png"/></div>
<br>

## 3  Operating Evironment
* ubuntu 16.04
* CUDA8.0 & CUdnn6.0
>这里环境的配置可以参考ＣＳＤＮ上的文章
  * ```sudo pip install CUDA&cudnn```
* opencv3 for python2.7
  * ```sudo pip install opencv3```
* tensorflow 1.6
  * ```sudo pip install tensorflow-gpu ```

## 4 Data Peparation
### 4.1 Clone the code to the local：<br>
```Python
git clone https://github.com/ZGX010/Classify_road_image.git
```
### 4.2 Operating environment test
Run detection in the file directory
```Ｐython
python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```

### 4.3 Handling image data processing formats that require training
* 如果你只是想验证一下，可以只使用我提供的数据集．数据和InceptionV4的预训练模型放置的地址为＇./tmp/data/＇,由于该文件夹太大超出了上传文件的大小限制，所以我将他们单独放置在网盘上，如果你需要可以按照文件中的readme操作获得数据与模型的下载链接．其中mydata文件夹中包含按类别分别放置的道路影像数据． <br>
脚本Ｃonvert_data.py是通过slim自带的download_and_convert_fllower.py修改而来．如果需要验证请直接运行． <br>
```python
convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```
* 而如果你想要训练自己的数据集，使网络在你的数据上表现的更好，需要进行如下修改，其中包括修改验证集的图片数量/划分训练集和验证集的数量/最后输出的类别数量/在数据集字典中添加数据的名字．
> 修改Ｃonvert_mydata文件，参数_NUM_SHARDS是指将训练的数据打包成几个部分，如果设置为２，那么脚本将会把数据集文件中的图片按训练和验证分别打包成２个ＴＦＲecord文件．参数_NUM_VALIDATION指定了从图像中抽取多少张影像作为验证集，如果你设置它为１５０，那么将会有１５０张影像被用于测试模型的准确率，
<br>
```Python
_NUM_SHARDS = ？ #class of file
_RANDOM_SEED = 4
_NUM_VALIDATION = ？ #number of the validation class
```
<br>
> 由于convert_mydata引用了mydata所以我们需要在mydata中详细划分用于训练的样本数量和用于验证的样本数量，同时也可以修改打包文件的文件名，并设定输出结果的类别数量.如果你训练的网络最后输出有16种，你可以在_NUM_CLASSES处设置为16，当你有空这一类时也要将其视为一类．
<br>
> Modify the mydata.py file
```python
SPLITS_TO_SIZES = {'train': ？, 'validation': ？ }
_FILE_PATTERN = 'mydata_%s_*.tfrecord'
_NUM_CLASSES = ？
```
> 由于mydata文件引用了dataset_factry,所以需要先将mydata引入dataset_factry，并在datasets_map中注册数据．
> Modify the dataset_factry.py file
```python
from datasets import mydata
```
```python
datasets_map = {
    'mydata': mydata,}
```

## 5  Train model

```Python
python train_image_classifier.py \
--train_dir='./tmp/data/mydata/train_logs' \
--dataset_name=mydata \
--dataset_split_name=train \
--dataset_dir=./tmp/data/mydata
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits 
```
<br>

## 6  Eval model
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

## 7  Export model
### 7．1 Export model graph
```Python
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

### 7.2 Load trained parameters for the graph architecture
```Python
python freeze_graph.py \
--input_graph=./tmp/data/mydata/inception_v4_inf_graph.pb \
--input_checkpoint=./tmp/data/mydata/train_logs/model.ckpt-51498 \
--input_binary=true \
--output_node_names=InceptionV4/Logits/Predictions \
--output_graph=./tmp/data/mydata/crake_classify.pb
```
<br>

## 8  Load the model and inference
```Python
python classify_image_inception_v3.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

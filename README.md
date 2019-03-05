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
> 如果你只是想验证一下，可以只使用我提供的数据集．数据和InceptionV4的预训练模型放置的地址为＇./tmp/data/＇,由于该文件夹太大超出了上传文件的大小限制，所以我将他们单独放置在网盘上，如果你需要可以按照文件中的readme操作获得数据与模型的下载链接．其中mydata文件夹中包含按类别分别放置的道路影像数据． <br>
脚本Ｃonvert_data.py是通过slim自带的download_and_convert_fllower.py修改而来．如果需要验证请直接运行． <br>
```python
convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```
> 而如果你想要训练自己的数据集，使网络在你的数据上表现的更好，需要进行如下修改，其中包括修改验证集的图片数量/划分训练集和验证集的数量/最后输出的类别数量/在数据集字典中添加数据的名字．
* 修改Ｃonvert_mydata文件，参数_NUM_SHARDS是指将训练的数据打包成几个部分，如果设置为２，那么脚本将会把数据集文件中的图片按训练和验证分别打包成２个ＴＦＲecord文件．参数_NUM_VALIDATION指定了从图像中抽取多少张影像作为验证集，如果你设置它为１５０，那么将会有１５０张影像被用于测试模型的准确率，
<br>

```Python
_NUM_SHARDS = ？ #class of file
_RANDOM_SEED = 4
_NUM_VALIDATION = ？ #number of the validation class
```
<br>
* 由于convert_mydata引用了mydata所以我们需要在mydata中详细划分用于训练的样本数量和用于验证的样本数量，同时也可以修改打包文件的文件名，并设定输出结果的类别数量.如果你训练的网络最后输出有16种，你可以在_NUM_CLASSES处设置为16，当你有空这一类时也要将其视为一类．
<br>
```python
SPLITS_TO_SIZES = {'train': ？, 'validation': ？ }
_FILE_PATTERN = 'mydata_%s_*.tfrecord'
_NUM_CLASSES = ？
```
<br>
* 由于mydata文件引用了dataset_factry,所以需要先将mydata引入dataset_factry，并在datasets_map中注册数据．
<br>

```python
from datasets import mydata
```
```python
datasets_map = {
    'mydata': mydata,}
```

## 5  Train model
在train_image_classifier文件中可以设置参数包括：优化器参数/学习率参数/数据集参数/和迁移学习参数． <br>
其中优化器默认是Ｒmsprop，而可选择的有＂adadelta＂/"adagrad"/"adam"/"ftrl"/"momentum"/"sgd"/"rmsprop".你可以在脚本中直接修改对应的名称，也可以将脚本修改为Ｎone，在执行时将参数传入． <br>
学习率衰减类型默认是＂exponetial＂,当然你也可以选择其他的，脚本里对应位置都给出了提示．如果你的网络在训练刚开始时loss下降非常的缓慢，那么你可以升高learning_rate为网络初始化一个较高的学习率帮助网络快速收敛．但是如果你的网络训练的后期loss一直起伏不定，那么你可以修改end_learning_rate更小来尽可能的寻找一个最优结果． <br>
在数据集参数中，默认训练的对象为数据集mydata，默认训练的模型为InceptionV4,如果你需要训练其他网络可以在脚本中修改，batchsize设定了每次训练输入网络的图像数量，如果你的ＧＰＵ并不大请改小此处．max_number_of_steps设定了训练的最大步数，脚本会根据初始学习率与最终学习率以及最大步数计算每一步参数移动的距离．
<br>

```Python
python train_image_classifier.py \
--train_dir='./tmp/data/mydata/train_logs' \
--dataset_name=mydata \
--dataset_split_name=train \
--dataset_dir=./tmp/data/mydata
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits 
```
<br>
训练过程中你可以使用Tensorboard观察训练过程中loss是如何一步步下降，当所有的log文件都在mydata文件下时，就可以比较不同训练参数时loss的变化．
```Python
tensorboard --log_dir=./tmp/data/mydata/
```
<br>

## 6  Eval model
这里提供了两个评估脚本eval_image_classifier会计算网络在有验证集上的表现并输出准确率和前五类召回率，而predict将会输出验证集上每张图片的表现．
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

```Python
python predict.py \
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
export_inference_graph只能导出模型的结构并没有参数，这样做是为了保留网络结构，参数在可以变更．我们还会将模型的参数导入到网络，方便后来推理时加载． <br>
参数output_file设定了输出导出模型的位置与名字． <br>

```Python
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

freeze_graph将模型训练过程中任意一步下的参数加载进刚刚导出的模型结构中，其中input_graph是网络结构文件的位置，input_checkpoint是网络参数位置，output_graph是输出模型的位置．此时的pb模型包含了结构与参数．
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
在推理的脚本里，载入了刚刚生成的pb模型，并读取了./test文件夹中的每一个图片，这里你不需要考虑图像尺寸和通道的问题，我将影像转为了３通道，并将图片统一压缩为299＠299的尺寸．进行推理时将自动创建./yes和./no两个文件夹并将按照推理结果将图片放置进对应的文件夹，以实现分类的效果．后面我将会推出ＴensorRT的加速版本以实现80km/h以上的实时检测． <br>
```Python
python classify_image_inception_v4.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

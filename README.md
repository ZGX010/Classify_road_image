# Classify_road_image
## 1  Project Function：<br>
This project trains the Inception_V4 model on the road image to realize the classification of the road image according to the disease. The accuracy of the model on the test set is 0.95.　At the same time, I provide a model that has been trained on the disease image to facilitate fur－training　and shorten training time.　
<br>

The inference script will load the trained PB model and read the image from the specified folder. After entering the neural network, the script will calculate the scores of the two categories and move the image to the corresponding folder according to the score.　
<br>

The following GIF is the script running process．
<br>
<div align=center><img width="450" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

## 2  Inception_V4 model
The following figure shows the structure of the inceptionV4 model, which consists of model-a/model-b/model-c. If you want to know more about the model, you can read the related papers.The default image size of the model is 299＠299, which you can modify in the script.
<div align=center><img width="800" height="480" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png"/></div>
<div align=center><img width="800" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png"/></div>
<br>

## 3  Operating Evironment
* ubuntu 16.04
* CUDA8.0 & CUdnn6.0
  * link:
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
> If you just want to try training, you can just use the data set I provided.The address of the data and InceptionV4 pre-training model is ./tmp/data/, but the size of the folder exceeds the upload limit, so I placed them separately on the network disk. Please follow the readme in the mydata folder to get the download link. The mydata folder contains road image data placed by category. 
<br>
The script Convert_data.py is modified by download_and_convert_data.py. If you need verification, please run it directly. 
<br>

```python
convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```
> If you want to train your own dataset or make the network perform better on your dataset, you need to make the following changes, including modifying the number of images in the validation set/dividing the training set and the number of validation sets/the final output category / Add the name of the data in the dataset dictionary. 
<br>

* Modify the Convert_mydata file. The parameter　＇_NUM_SHARDS＇　refers to packing the training data into several. If set　＇_NUM_SHARDS＇ to ＇2＇, the script will package the images in the dataset file into 2 TFRecord files according to training and verification. The parameter ＇_NUM_VALIDATION＇ specifies how many images are extracted from the image as a validation set. If you set ＇_NUM_VALIDATION＇ to 150, then 150 images will be used to test the accuracy of the model.
<br>

```Python
_NUM_SHARDS = 2 #class of file
_RANDOM_SEED = 4
_NUM_VALIDATION = 150 #number of the validation class
```
<br>

* Since convert_mydata references mydata, you need to divide the number of samples for training and verification in mydata. You can also modify the file name of the packaged file and set the number of categories of output. If you train the network to output 16 results then Set ＇_NUM_CLASSES＇ to 16.
<br>

```python
SPLITS_TO_SIZES = {'train': 3000 , 'validation': 150 }
_FILE_PATTERN = 'mydata_%s_*.tfrecord'
_NUM_CLASSES = 2
```
<br>

* Modify the dataset_factry to register the mydata dataset.
<br>

```python
from datasets import mydata
```
<br>
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

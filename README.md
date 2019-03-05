# Classify_road_image

## 1  Project Function

This project trains the Inception_V4 model on the road image to realize the classification of the road image according to the disease. The accuracy of the model on the test set is 0.95.　At the same time, I provide a model that has been trained on the disease image to facilitate fur－training　and shorten training time.　
<br>

The inference script will load the trained PB model and read the image from the specified folder. After entering the neural network, the script will calculate the scores of the two categories and move the image to the corresponding folder according to the score.　
<br>

The following GIF is the script running process．
<br>
<div align=center><img width="450" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>
<br>

## 2  Inception_V4 model
The following figure shows the structure of the inceptionV4 model, which consists of model-a/model-b/model-c. If you want to know more about the model, you can read the related papers.The default image size of the model is 299＠299, which you can modify in the script.
<div align=center><img width="800" height="480" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png"/></div>
<div align=center><img width="800" height="240" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png"/></div>
<br>
<br>

## 3  Operating Evironment
* ubuntu 16.04
* CUDA8.0 & CUdnn6.0
  * link:
* opencv3 for python2.7
  * ```sudo pip install opencv3```
* tensorflow 1.6
  * ```sudo pip install tensorflow-gpu ```
<br>
<br>

## 4 Data Peparation
### 4.1 Clone the code to the local：<br>
```Python
git clone https://github.com/ZGX010/Classify_road_image.git
```
<br>

### 4.2 Operating environment test
Run detection in the file directory
```Ｐython
python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```
<br>

### 4.3 Processing training pictures
> If you just want to try training, you can just use the data set I provided.The address of the data and InceptionV4 pre-training model is ./tmp/data/, but the size of the folder exceeds the upload limit, so I placed them separately on the network disk. Please follow the readme in the mydata folder to get the download link. The mydata folder contains road image data placed by category. <br>
The script Convert_data.py is modified by download_and_convert_data.py. If you need verification, please run it directly. <br>
```python
convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```
<br>

> If you want to train your own dataset or make the network perform better on your dataset, you need to make the following changes, including modifying the number of images in the validation set/dividing the training set and the number of validation sets/the final output category / Add the name of the data in the dataset dictionary. <br>
* Modify the Convert_mydata file. The parameter　＇_NUM_SHARDS＇　refers to packing the training data into several. If set　＇_NUM_SHARDS＇ to ＇2＇, the script will package the images in the dataset file into 2 TFRecord files according to training and verification. The parameter ＇_NUM_VALIDATION＇ specifies how many images are extracted from the image as a validation set. If you set ＇_NUM_VALIDATION＇ to 150, then 150 images will be used to test the accuracy of the model.
```Python
_NUM_SHARDS = 2 #class of file
_RANDOM_SEED = 4
_NUM_VALIDATION = 150 #number of the validation class
```
* Since convert_mydata references mydata, you need to divide the number of samples for training and verification in mydata. You can also modify the file name of the packaged file and set the number of categories of output. If you train the network to output 16 results then Set ＇_NUM_CLASSES＇ to 16.
```python
SPLITS_TO_SIZES = {'train': 3000 , 'validation': 150 }
_FILE_PATTERN = 'mydata_%s_*.tfrecord'
_NUM_CLASSES = 2
```
* Modify the dataset_factry to register the mydata dataset.
```python
from datasets import mydata
...
datasets_map = {
    'mydata': mydata,}
```
<br>
<br>

## 5  Train model
Parameters that can be set in the train_image_classifier file include: optimizer parameters/learning rate parameters/dataset parameters/and fur－training parameters.
<br>

> The optimizer you can choose has "adadelta"/"adagrad"/"adam"/"ftrl"/"momentum"/"sgd"/"rmsprop", the default is Rmsprop. You can modify the corresponding name directly in the script or Modify the script to None and pass in the parameters when executed.
<br>

> You can choose a variety of learning rate attenuation types, the default is "exponetial". If your network is slow to slow down at the beginning of training, then you can raise the learning_rate to initialize a higher learning rate for the network to help the network converge quickly. If your network has been fluctuating in the late stages of training, then you need to reduce the end_learning_rate to find an optimal result.
<br>

> In the dataset parameters, the default training object is the mydata dataset. The default training model is InceptionV4. If you need to train other networks, you can modify them in the script. Batchsize sets the number of images input to the network each iteration. Max_number_of_steps sets the maximum number of steps to train. The script determines the distance of parameter movement in each step network by the initial learning rate and the final learning rate and the maximum number of steps.
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

During the training, you can use Tensorboard to observe how the loss is reduced step by step during the training. When the log files of different training parameters are under the mydata file, you can compare the changes of loss under different conditions.
<br>

```Python
tensorboard --log_dir=./tmp/data/mydata/
```
<br>
<br>

## 6  Eval model
Two evaluation scripts are provided here. eval_image_classifier.py calculates the performance of the network on the validation set and outputs the accuracy and top five recalls, while predict.py will output the performance of each image on the validation set.
<br>

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
<br>

## 7  Export model
### 7．1 Export model graph
Export_inference_graph.py only exports the structure of the model with no parameters. We will also import the trained parameters into the network and then perform the inference calculations. 
<br>

The parameter output_file sets the location and name of the output export model.　
<br>

```Python
python export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v4 \
--output_file=./tmp/data/mydata/inception_v4_inf_graph.pb \
--dataset_name=mydata
```
<br>

### 7.2 Load trained parameters for the graph architecture
Freeze_graph.py loads the parameters of any stage of the model into the exported network structure. Input_graph is the location of the network structure file. Input_checkpoint is the location of the network parameter. Output_graph is the location of the output model. The pb model at this time contains the structure and parameters.
<br>
```Python
python freeze_graph.py \
--input_graph=./tmp/data/mydata/inception_v4_inf_graph.pb \
--input_checkpoint=./tmp/data/mydata/train_logs/model.ckpt-51498 \
--input_binary=true \
--output_node_names=InceptionV4/Logits/Predictions \
--output_graph=./tmp/data/mydata/crake_classify.pb
```
<br>
<br>

## 8  Load the model and inference
The inference script loads the generated pb model and will read every picture in the './test' folder. You don't need to worry about image size and channel, because I convert the image to 3 channels and compress the image into 299@299 size. When you make a reasoning, you will automatically create two folders './yes' and './no' and move the picture to the corresponding folder according to the inference result. In the future, I will launch an accelerated version of TensorRT to achieve real-time detection above 80km/h. <br>
```Python
python classify_image_inception_v4.py \
--model_path=./tmp/data/mydata/crake_classify.pb \
--label_path=./tmp/data/mydata/labels.txt \
--image_dir=./test
```

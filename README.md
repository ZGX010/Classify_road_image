# Classify_road_image
## 1  Project Function：<br>
This project trains the Inception_V4 model on the road image to realize the classification of the road image according to the disease. The accuracy of the test set is:0.95． At the same time, I provide a model that has been trained on the disease image to facilitate further training. Apply to your own data and shorten training time.<br>
When you run the [8] inference script, the trained PB model will be loaded, and Tensorflow will read the image from the specified folder and enter the neural network to calculate the scores for the two categories. Move the image to the corresponding folder based on the score<br>
The following GIF is the script running process．<br>
<div align=center><img width="520" height="320" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/classimage.gif"/></div>
<br>

## 2  Inception_V4 model

<div align=center><img width="1000" height="629" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4.png"/></div>
<div align=center><img width="1000" height="337" src="https://github.com/ZGX010/Classify_road_image/blob/master/doc/inceptionv4model.png"/></div>
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

Only train the data provided by this project: convert the data that needs to be trained into the TFRecord format, because the mydata dataset has been added to the script, so the dataset that needs to be trained is directly named mydata.
```python
convert_data.py \
--dataset_name=mydata \
--dataset_dir=./tmp/data/mydata
```
Train your own data set：
Modify the covert_mydata.py file
```Python
_NUM_SHARDS = ？ #class of file
_RANDOM_SEED = 4
_NUM_VALIDATION = ？ #number of the validation class
```
Modify the mydata.py file
```python
SPLITS_TO_SIZES = {'train': ？, 'validation': ？ }
_FILE_PATTERN = 'mydata_%s_*.tfrecord'
_NUM_CLASSES = ？
```
Modify the dataset_factry.py file
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

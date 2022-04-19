# Predicting the P-glycoprotein inhibitor

Predicting the P-glycoprotein inhibitor with Tensorflow.

## Table of Contents

- [Decripstion of folder and file in the repository](#decripstion-of-folder-and-file-in-the-repository)
- [Implementation steps](#implementation-steps)
    - [Install dependencies](#install-dependencies)
    - [Check device](#check-device)
    - [Preprocessing](#preprocessing)
    - [Training](#training)
    - [Predict](#predict)
- [Support Command-line](#support-command-line)

## Decripstion of folder and file in the repository
- **bin** folder includes the source code and requirements.txt.
- **dataset** includes the Viet Nam Traffic Sign data.
- **output** includes:

    - **train_model.png** - image after select model name.
    - **predict_model.png** - image when load model.
    - **best_model.h5** file.
    - **training_accuracy.png** - training accuracy metrics image.
    - **acc_loss_history.json** - training accuracy and loss metrics.
    - **images** folder - visualize feature maps images of model sample.
    - checkpoints when training model.
## Implementation steps

### Install dependencies
```console
pip install -r .\bin\requirements.txt
```
### Check device

- Get CPU and GPU information.
- Check Graphviz software.
```console
py .\bin\check_device.py
```

Output:
```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 10256441937581857634
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11320098816
locality {
  bus_id: 1
  links {
  }
}
incarnation: 4824839391425357507
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
xla_global_id: 416903419
]
Great! Graphviz already exists.
```

> If Graphviz does not exists, go [there](https://graphviz.org/download/) to install software. 

### Preprocessing

Creating the P-glycoprotein Inhibitor dataset to `dataset` folder.

#### The process

**Functions**:
- **str_list(string)** : convert string to list numeric.
- **preprocessing_image_pipeline(image_size)** :  preprocessing image pipeline with **image_size**.
- **download_dataset(file_id, dest_path)**: download file from google drive with `id`=**file_id** to **dest_path**.
- **unzip_dataset(zip_path, dest_path, pwd=None)**: unzip **zip_path** file with password = **pwd** to **dest_path**.
- **make_folder(parents_path, folder_name)** : create folder.
- **create_image(image_path, name, smile)** : create **name**.png image from **smile** and save to **image path**.

### Training

- Training model with KFold.
- Save training accuracy metrics as json file.

**Usage**:

```console
train.py [-h] [--bs BS] [--isize ISIZE ISIZE]
              [--epos EPOS]
              [--model {keras,simple,vgg16,mobilenetv2}]

Train the P-glycoprotein Inhibitor.
```

**Options** :

- `-h` `--help`: Show this message and exit.
- `--bs` `--batch_size`: **integer** - Batch size for dataset (default: 128).
- `--isize` `--image_size`: **integer** - Image size (default: (224, 224)).
- `--epos` `--epochs`: **integer** - Epochs for training (default: 2).
- `--model`: Model name as *keras/simple/vgg16/mobilenetv2* (default: **mobilenetv2**).


#### The process

**Example**:
- Training **mobilenetv2** model with **batch size = 64**, **image size = (256,256)** and **2 epochs**.
```console
py .\bin\train.py --bs 64 --isize 224 224 --epos 2
```

Output:
```
P-gp_act_inact_dataset.csv already exists!
Loading dataset...
Done!
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9412608/9406464 [==============================] - 0s 0us/step
9420800/9406464 [==============================] - 0s 0us/step
Model: "simple"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 1)]     0         
                                                                 
 sequential (Sequential)     (None, 224, 224, 1)       0         
                                                                 
 rescaling (Rescaling)       (None, 224, 224, 1)       0         
                                                                 
 conv2d (Conv2D)             (None, 75, 75, 32)        320       
      ....................................................     
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1024)              132096    
                                                                 
 activation_9 (Activation)   (None, 1024)              0         
                                                                 
 batch_normalization_9 (Batc  (None, 1024)             4096      
 hNormalization)                                                 
                                                                 
 dropout_5 (Dropout)         (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 1)                 1025      
                                                                 
=================================================================
Total params: 555,265
Trainable params: 552,001
Non-trainable params: 3,264
_________________________________________________________________
Training simple model with batch size = 128, image size = (224, 224) and epochs = 2.
Epochs 1/2
Training Fold 1
24/24 [==============================] - 18s 220ms/step - loss: 0.8955 - accuracy: 0.4860 - recall: 0.5053 - precision: 0.4866 - val_loss: 0.6932 - val_accuracy: 0.5000 - val_recall: 1.0000 - val_precision: 0.5000
Training Fold 2
24/24 [==============================] - 5s 190ms/step - loss: 0.8455 - accuracy: 0.4937 - recall: 0.5160 - precision: 0.4940 - val_loss: 0.6932 - val_accuracy: 0.5000 - val_recall: 1.0000 - val_precision: 0.5000
Training Fold 3
24/24 [==============================] - 5s 188ms/step - loss: 0.8099 - accuracy: 0.5040 - recall: 0.5173 - precision: 0.5039 - val_loss: 0.6932 - val_accuracy: 0.5000 - val_recall: 1.0000 - val_precision: 0.5000
Training Fold 4
24/24 [==============================] - 4s 186ms/step - loss: 0.8046 - accuracy: 0.5033 - recall: 0.5140 - precision: 0.5033 - val_loss: 0.6936 - val_accuracy: 0.5000 - val_recall: 0.0000e+00 - val_precision: 0.0000e+00
Training Fold 5
24/24 [==============================] - 5s 207ms/step - loss: 0.8015 - accuracy: 0.5023 - recall: 0.5219 - precision: 0.5022 - val_loss: 0.6934 - val_accuracy: 0.5000 - val_recall: 0.0000e+00 - val_precision: 0.0000e+00
Epochs 2/2
Training Fold 1
24/24 [==============================] - 5s 219ms/step - loss: 0.7907 - accuracy: 0.5040 - recall: 0.4993 - precision: 0.5040 - val_loss: 0.6947 - val_accuracy: 0.5000 - val_recall: 0.0000e+00 - val_precision: 0.0000e+00
Training Fold 2
24/24 [==============================] - 5s 190ms/step - loss: 0.7760 - accuracy: 0.4953 - recall: 0.5020 - precision: 0.4954 - val_loss: 0.6938 - val_accuracy: 0.4801 - val_recall: 0.6410 - val_precision: 0.4849
Training Fold 3
24/24 [==============================] - 5s 188ms/step - loss: 0.7599 - accuracy: 0.5106 - recall: 0.5040 - precision: 0.5108 - val_loss: 0.6955 - val_accuracy: 0.5000 - val_recall: 1.0000 - val_precision: 0.5000
Training Fold 4
24/24 [==============================] - 5s 190ms/step - loss: 0.7708 - accuracy: 0.4970 - recall: 0.4947 - precision: 0.4970 - val_loss: 0.6939 - val_accuracy: 0.4880 - val_recall: 0.5213 - val_precision: 0.4888
Training Fold 5
24/24 [==============================] - 5s 194ms/step - loss: 0.7581 - accuracy: 0.5050 - recall: 0.4940 - precision: 0.5051 - val_loss: 0.6951 - val_accuracy: 0.4960 - val_recall: 0.0426 - val_precision: 0.4571
<Figure size 800x800 with 2 Axes>
```

### Predict

**Usage**:

```console
predict.py [-h] [--model MODEL]

Predict the P-glycoprotein Inhibitor.
```

**Options** :

- `-h` `--help`: Show this message and exit.
- `--model`: Model path want to use (default: **best_model.h5**).

#### The process

- Run the process.
```console
py .\bin\predict.py
```

Output:
```
Model: "simple"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 1)]     0         
                                                                 
 sequential (Sequential)     (None, 224, 224, 1)       0         
                                                                 
 rescaling (Rescaling)       (None, 224, 224, 1)       0         
                                                                 
 conv2d (Conv2D)             (None, 75, 75, 32)        320       
      ....................................................     
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1024)              132096    
                                                                 
 activation_9 (Activation)   (None, 1024)              0         
                                                                 
 batch_normalization_9 (Batc  (None, 1024)             4096      
 hNormalization)                                                 
                                                                 
 dropout_5 (Dropout)         (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 1)                 1025      
                                                                 
=================================================================
Total params: 555,265
Trainable params: 552,001
Non-trainable params: 3,264
_________________________________________________________________
None
---
Activity
---
Inactivate
---
Activity
---
Inactivate
---
Inactivate
---
Activity
---
Activity
---
Inactivate
---
Activity
```


## Support Command-line

- To set your global username/email configuration:
Open the command line.

```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

- Create Virtual Environment :

```console
python -m venv venv
venv\Scripts\activate
(venv) >
``` 
> fix `cannot be loaded because running scripts is disabled on this system`:
```console
set-ExecutionPolicy RemoteSigned -Scope CurrentUser 
Get-ExecutionPolicy
Get-ExecutionPolicy -list  
```
 
- Create requirements.txt :

```console
py -m pipreqs.pipreqs . --encoding=utf8
``` 

- Create requirements.txt with all library already exists:

```console
pip freeze > requirements.txt
```

- Install requirements:
```console
pip install -r requirements.txt
```
- Use pytest:
```console
pytest tests
pytest tests/test_export_data.py::TestRealTime
pytest tests/test_export_data.py::TestRealTime::test_real_time_negative
```
- Upgrade library:

```console
pip install -U <library>
```
- Upgrade pip:

```console
python -m pip install --upgrade pip
```
- Delete all commit history in Github:
  - Checkout
    ```consolse
    git checkout --orphan latest_branch
    ```
  - Add all the files
    ```consolse
    git add -A
    ```
  - Commit the changes
    ```consolse
    git commit -am "commit message"
    ```
  - Delete the branch
    ```consolse
    git branch -D main
    ```
  - Rename the current branch to main
    ```consolse
    git branch -m main
    ```
  - Finally, force update your repository
    ```consolse
    git push -f origin main   
    ```
    PS: this will not keep your old commit history around
###### [on top](#table-of-contents)
# 'halp' Team Submission for USYD Challenge

## Install

To run our ensemble, simply run the `predict.py` file as shown below. The submission file will be output in the directory this github repo is saved down to.

```bash
$ python predict.py /path/to/input/files
```

## Hardware and services we tested and used

- Azure Notebooks - too slow and no GPU support
- Google Colaboratory Notebooks - used this to test and prototype model structures since Google generously provides GPU support
- Google Cloud AI Platform - experimented with this; however, strangely Google does not provide GPU access for accounts using their free trial
- Kaggle Kernel - used this to test and prototype; however, performance is inferior to Colaboratory
- Google Cloud Storage - used this to host datasets and files for our models since it allowed for easy integration with Colaboratory
- Nvidia GTX1060 GPU - used this to prototype small networks locally at home
- Nvidia RTX2080Ti GPU - on the last few days we gained access to a server with a RTX2080Ti GPU and trained our remaining models here


## Things we tried

### Image preprocessing

Image preprocessing involved the following steps.
1. Crop excess black pixels/space (usually on the left and right of the fundoscope image) by detecting the image's contours using OpenCV's `findContours` method.
2. Detect the radius and centre of the fundoscope's circle using OpenCV's `HoughCircles` function.
3. Pad the image accordingly so the circle detected in step 2 is centred.

The result is an image that maintains the original aspect ratio and ensures the circle is centre.
![Imgur](https://i.imgur.com/n0pLj0a.png)

We also experimented with the following preprocessing filters:

- Gaussian Filter
```python
def subtract_gaussian_blur(img):
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)
```
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
```python
def get_clahe_img(img, clip_limit, grid_size):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.uint8)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img
```
- Greyscale Images

However, we settled with standard RGB images since we didn't observe significant improvements using the above filters.

Since we used transfer learning, we also preprocessed images according to what the models were original trained on. For example, for DenseNet121, we mean-centred all images

- Floating Point

To conserve RAM on our earlier setups we decided to convert the data to `float16` so the bulk of our work was doing in `float16`. However, towards the end of the competition, with a setup which no longer had such RAM restrictions we noticed that `float32` performed slightly better for an equivalent dataset and model so in hindsight we should have ported all our earlier `float16` work to `float32` to potentially achieve greater results 

### Image sizes we tried
We experimented with image height and widths of 256, 384 and 512 pixels and observed a positive relationship between image size and our model's predictive power so we settled on using images of 512 pixels in height and width.

### Models we experimented with

- Xception
- VGG16
- ResNet18
- ResNet34
- ResNet50
- InceptionV3
- MobileNetV2
- DenseNet121

### Models we used in our final ensemble

- DenseNet121
- 

### Model structures we experimented with

Given we opted to use pre-trained ImageNet models, we decided to experiment with the structure of the model following the pre-trained network in an attempt to search for a gain in performance.

#### Standard global average pooling layer
```
# These are the layers following the pre-trained model e.g. densenet model would be here
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
# model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))
```
#### Concatenating average and max pooling layers
Here we are concatenating average and max pool layers to get the best of both worlds. In order to achieve this, we used Keras' functional API.
```python
# These are the layers following the pre-trained model e.g. densenet model would be here
pool_1 = GlobalAveragePooling2D()(densenet.output)
pool_2= GlobalMaxPooling2D()(densenet.output)
concat_pool = concatenate([pool_1, pool_2])
bat_1 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(concat_pool)
drop_1 = Dropout(0.5)(bat_1)
linear_1 = Dense(1024, activation = 'relu')(drop_1)
bat_2 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(linear_1)
drop_2 = Dropout(0.5)(bat_2)
final = Dense(NUM_CLASSES, activation = 'sigmoid')(drop_2)
model = Model(inputs = densenet.input, outputs = final)
```
#### Just using average pooling layer
Here we are concatenating average and max pool layers to get the best of both worlds. In order to achieve this, we used Keras' functional API.
```python
# These are the layers following the pre-trained model e.g. densenet model would be here
pool_1 = GlobalAveragePooling2D()(densenet.output)
bat_1 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(pool_1)
drop_1 = Dropout(0.5)(bat_1)
linear_1 = Dense(1024, activation = 'relu')(drop_1)
bat_2 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(linear_1)
drop_2 = Dropout(0.5)(bat_2)
final = Dense(NUM_CLASSES, activation = 'sigmoid')(drop_2)
model = Model(inputs = densenet.input, outputs = final)
```
#### Just using max pooling layer
Here we are concatenating average and max pool layers to get the best of both worlds. In order to achieve this, we used Keras' functional API.
```python
# These are the layers following the pre-trained model e.g. densenet model would be here
pool_1 = GlobalMaxPooling2D()(densenet.output)
bat_1 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(pool_1)
drop_1 = Dropout(0.5)(bat_1)
linear_1 = Dense(1024, activation = 'relu')(drop_1)
bat_2 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(linear_1)
drop_2 = Dropout(0.5)(bat_2)
final = Dense(NUM_CLASSES, activation = 'sigmoid')(drop_2)
model = Model(inputs = densenet.input, outputs = final)
```


### Class types we experimented with

We experimented with class types by representing the problem as a multi-class, regression and multi-label problem and observed our model performed better when the problem was represented as a multi-label problem. This can be justified by the fact that certain features in more severe cases of diabetic retinopathy may also be present in less severe cases. Below is how we encoded each label as a multi class

```bash
Label: 0 => [1, 0, 0, 0, 0]
Label: 1 => [1, 1, 0, 0, 0]
Label: 2 => [1, 1, 1, 0, 0]
Label: 3 => [1, 1, 1, 1, 0]
Label: 4 => [1, 1, 1, 1, 1]
```

### Loss functions we experimented with

 Since we opted to represent the problem as a multi-label problem, we used the binary cross entropy loss function to optimise our model. We also experimented with our own custom F1 loss function (below). 
 
```python
def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
```
Below is our attempt to optimise using quadratic weighted kappa with mixed results.

```
def kappa_loss(predictions, labels, y_pow=1, eps=1e-15, num_ratings=5, batch_size=10, name='kappa'):
    with tf.name_scope(name):
        labels = tf.to_float(labels)
        repeat_op = tf.to_float(
            tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((num_ratings - 1)**2)

        pred_ = predictions**y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [batch_size, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(labels, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), labels)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings]))
                / tf.to_float(batch_size))
        try:
            return -(1 - nom / denom)
        except Exception:
            return -(1 - nom / (denom + eps))
```
Our experiments showed optimising on F1-score yielded similar results to binary cross entropy and our attempt to optimise using quadratic weighted kappa was a failure. Therefore, we decided to use Keras' inbuilt binary cross entropy function to optimise our model.

### Optimizers we experimented with
- Stochastic Gradient Descent (with and without momentum)
- Adam
- Adagrad
- AdaBoost
- RMSProp

From our experiments we concluded that SGD typically converged to a more globally optimal solution than dynamic optimizers like Adam.

### Freezing

We also experimented with a two-stage training process by freezing the pre-trained models first while training the latter layers. Once the latter layers converged, we then trained the entire model unfrozen. After comparing results, we opted against this approach as we didn't observe a significant improvement in accuracy.


### Learning rate schedule

We experimented with constant learning rates ranging from 0.1 to 0.00001 as well as scheduled learning rates like the one below. We settled on the following learning rate schedule since our solution uses models with pre-trained weights, justifying our lower initial learning rate.

```python
def lr_schedule(epoch):
    lr = 0.0003
    if epoch > 25:
        lr = 0.00015
    if epoch > 30:
        lr = 0.000075
    if epoch > 35:
        lr = 0.00003
    if epoch > 40:
        lr = 0.00001
    return lr
```



### ImageDataGenerator
We used the ImageDataGenerator class in Keras to generate random transformations of our training dataset by rotating and flipping the images to prevent our model from overfitting. We opted to avoid cropping or zooming transformations given our test data would also be preprocessed in a manner where  the fundoscope would be centred and the original aspect ratio would be maintained.

```python
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=360
)
```

### Test Time Augmentation (TTA)
We also experimented with test time augmentation (below); however, experienced worse results using it and therefore opted against including it in our final model.

```python
def model_predict(x_test, path_to_model, batch_size=10):
    model = load_model(path_to_model, custom_objects={'f1_loss': f1_loss, 'multi_label_acc': multi_label_acc, 'f1_m': f1_m})
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360
    )
    tta_steps = 5
    predictions = []
    for i in range(tta_steps):
        preds = model.predict_generator(datagen.flow(x_test, batch_size=batch_size, shuffle=False, seed=1), steps=len(x_test) / batch_size)
        predictions.append(preds)
    pred = np.mean(np.asarray(predictions), axis=0)
    return pred > 0.5
```


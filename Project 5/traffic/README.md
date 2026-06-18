# Model Testing Document

## First Parameters Test:
I used the same parameters used in the lecture:
```python
model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
```
Results:
``` bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 17s 26ms/step - accuracy: 0.0515 - loss: 6.9758  
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.0554 - loss: 3.5941      
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 20s 39ms/step - accuracy: 0.0554 - loss: 3.5458  
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 17s 31ms/step - accuracy: 0.0554 - loss: 3.5232      
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 16s 31ms/step - accuracy: 0.0554 - loss: 3.5127 
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 26ms/step - accuracy: 0.0539 - loss: 3.5077 
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 26ms/step - accuracy: 0.0543 - loss: 3.5053     
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 26ms/step - accuracy: 0.0554 - loss: 3.5041 
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 17s 34ms/step - accuracy: 0.0554 - loss: 3.5034     
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.0554 - loss: 3.5031 
333/333 - 3s - 9ms/step - accuracy: 0.0576 - loss: 3.4915
```
A resoundingly bad result, with an accuracy of 5.76% on the test set.

## Second Parameters Test:
This time, I added a second convolutional layer, and increased the number of filters to 64:
```python
model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
```
Results:
``` bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 69s 133ms/step - accuracy: 0.4696 - loss: 2.8168    
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 66s 132ms/step - accuracy: 0.7725 - loss: 0.8260 
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 65s 130ms/step - accuracy: 0.8418 - loss: 0.5524 
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 72s 143ms/step - accuracy: 0.8840 - loss: 0.4113 
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 64s 129ms/step - accuracy: 0.8989 - loss: 0.3723 
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 64s 129ms/step - accuracy: 0.9117 - loss: 0.3185 
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 69s 137ms/step - accuracy: 0.9238 - loss: 0.2841 
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 69s 137ms/step - accuracy: 0.9358 - loss: 0.2227 
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 68s 137ms/step - accuracy: 0.9424 - loss: 0.2152 
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 56s 111ms/step - accuracy: 0.9421 - loss: 0.2204 
333/333 - 8s - 23ms/step - accuracy: 0.9649 - loss: 0.1663
```
Now we're talking :D, the accuracy is a 96.49%, a much better result than the previous and on par with the results in the project description, but the training time is much higher, although this is expected as I am using CPU instead of GPU.

## Third Parameters Test:
This time, I experimented with decreasing the dropout rate to 0.3 to see what would happen (Note, I also used Google Colab instead of my PC)
```python
model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
```

Results:
``` bash
500/500 ━━━━━━━━━━━━━━━━━━━━ 48s 94ms/step - accuracy: 0.6533 - loss: 1.7565
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 47s 93ms/step - accuracy: 0.8943 - loss: 0.4107
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 46s 93ms/step - accuracy: 0.9292 - loss: 0.2575
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 85s 98ms/step - accuracy: 0.9424 - loss: 0.2167
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 80s 94ms/step - accuracy: 0.9583 - loss: 0.1505
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 82s 93ms/step - accuracy: 0.9513 - loss: 0.1777
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 46s 93ms/step - accuracy: 0.9633 - loss: 0.1372
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 46s 93ms/step - accuracy: 0.9645 - loss: 0.1397
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 47s 93ms/step - accuracy: 0.9624 - loss: 0.1474
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 82s 94ms/step - accuracy: 0.9675 - loss: 0.1194
333/333 - 9s - 27ms/step - accuracy: 0.9745 - loss: 0.1303
```
The accuracy is now 97.45%, a slight improvement over the previous test, so a dropout rate of 0.5 was too high for this model, we will ignore the training time as I was using Google Colab.

## Fourth Parameters Test:
This time, I moved the second convolutional layer after the max pooling layer, and decreased the number of filters of the first layer to 32 filters to see if it would make a difference:
```python
model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
              64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"), 
            keras.layers.Dropout(0.3),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
```
Results:
``` bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 26ms/step - accuracy: 0.6532 - loss: 1.7756
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 26ms/step - accuracy: 0.9164 - loss: 0.3116
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 26ms/step - accuracy: 0.9432 - loss: 0.2037
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 21s 27ms/step - accuracy: 0.9544 - loss: 0.1676
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9676 - loss: 0.1168
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - accuracy: 0.9629 - loss: 0.1416
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 20s 28ms/step - accuracy: 0.9715 - loss: 0.1033
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 20s 27ms/step - accuracy: 0.9647 - loss: 0.1513
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9743 - loss: 0.1021
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 27ms/step - accuracy: 0.9751 - loss: 0.0987
333/333 - 3s - 10ms/step - accuracy: 0.9677 - loss: 0.1914
```
While the accuracy is slightly lower, the computation time is much lower, which makes sense as the number of filters before the pooling layer is lower, and the pooling layer reduces the input size for the second convolutional layer.
'''Concrete crack image classification'''
#%%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import matplotlib.pyplot as plt 
import os,datetime
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
#%%
#2 Data Loading
main_folder="C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment3\\Concrete Crack Images"
# Set the paths for the train and test folders where the split images will be saved

train_folder = 'C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment3\\train'
test_folder = 'C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment3\\test'
#%%# Set the train-test split ratio
train_ratio = 0.8

# Create the train and test folders if they don't already exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
#%%
# Iterate over each class (negative and positive) within the main folder
for class_folder in os.listdir(main_folder):
    class_path = os.path.join(main_folder, class_folder)
    if os.path.isdir(class_path):
        # Get the list of image files within the class folder
        image_files = os.listdir(class_path)
        
        # Split the image files into train and test sets
        train_files, test_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
        
        # Move the train files to the train folder
        for train_file in train_files:
            src_path = os.path.join(class_path, train_file)
            dest_path = os.path.join(train_folder, class_folder, train_file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)
        
        # Move the test files to the test folder
        for test_file in test_files:
            src_path = os.path.join(class_path, test_file)
            dest_path = os.path.join(test_folder, class_folder, test_file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)
# %%
file_path=r"C:\Users\IT\Desktop\AI lesson\DEEP LEARNING\hands-on\Assessment3"
train_path = os.path.join(file_path,"train")
test_path = os.path.join(file_path,"test")

BATCH_SIZE = 32
IMG_SIZE = (224,224)
train_dataset = keras.utils.image_dataset_from_directory(train_path,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
test_dataset = keras.utils.image_dataset_from_directory(test_path,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
#%%
#Take first batch of test data as the test dataset, the rest will be validation dataset
val_dataset = test_dataset.skip(1)
test_dataset = test_dataset.take(1)
# %%
#3. Convert the datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
#4. Create the data augmentation model
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
# %%
#5. Create the input preprocessing layer
preprocess_input = applications.mobilenet_v2.preprocess_input
# %%
#6. Apply transfer learning
class_names = train_dataset.class_names
nClass = len(class_names)
#(A) Apply transfer learning to create the feature extractor
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV3Large(input_shape=IMG_SHAPE,include_top=False,weights="imagenet",include_preprocessing=False)
base_model.trainable = False
# %%
#(B) Create the classifier
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass,activation='softmax')
# %%
#7. Link the layers together to form the model pipeline using functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
# x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)

# %%
#8. Compile the model
cos_decay = optimizers.schedules.CosineDecay(0.0005,50)
optimizer = optimizers.Adam(learning_rate=cos_decay)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#9. Evaluate the model before training
loss0,acc0 = model.evaluate(pf_test)
print("----------------Evaluation Before Training-------------------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)
# %%
#10. Create tensorboard
base_log_path = r"tensorboard_logs\concrete-crack"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#11. Model training
#Implement the EarlyStopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=3)
EPOCHS = 10
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb, early_stopping])
# %%
#12. Follow-up training
base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False
base_model.summary()
plot_model(base_model,show_shapes=True,show_layer_names=True)
#%%
#13. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#14. Evaluate the model after training
test_loss, test_acc = model.evaluate(pf_test)
print("----------------Evaluation After Training---------------")
print("Test loss = ",test_loss)
print("Test accuracy = ",test_acc)
# %%
#15. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)
#Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))
# %%
print(label_vs_prediction)

# %%
#16. Save the model
save_path = os.path.join("save_model","concrete_crack_classification_model.h5")
model.save(save_path)
# %%

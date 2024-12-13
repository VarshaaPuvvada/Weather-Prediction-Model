# Standard libraries
import os
import shutil
import random
from glob import glob

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.misc
import tensorflow as tf

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Reshape, Dense, Dropout, Flatten,
    LSTM, Input, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, callbacks, backend as K
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)

# Define the folder path
folder_path = '/content/gdrive/My Drive/dataset2'

# Check if the directory exists
if os.path.isdir(folder_path):
    # List the contents of the folder
    files = os.listdir(folder_path)
    print("Contents of 'dataset2':")
    for file in files:
        print(file)
else:
    print("The folder 'dataset2' does not exist at the specified path.")



# Define the base folder path
base_folder = '/content/gdrive/My Drive/dataset2'

# Collect image names from each category
try:
    cloudy = os.listdir(os.path.join(base_folder, 'cloudy'))
    rain = os.listdir(os.path.join(base_folder, 'rain'))
    sunrise = os.listdir(os.path.join(base_folder, 'sunrise'))
    sunshine = os.listdir(os.path.join(base_folder, 'shine'))

    print("Number of images in each category:")
    print(f"Cloudy: {len(cloudy)} images")
    print(f"Rain: {len(rain)} images")
    print(f"Sunrise: {len(sunrise)} images")
    print(f"Sunshine: {len(sunshine)} images")

except FileNotFoundError as e:
    print(f"Error: {e}")


#cloudy images
# Define the path to the 'cloudy' images in Google Drive
cloudy_folder = '/content/gdrive/My Drive/dataset2/cloudy'

# Display random images from the 'cloudy' folder
plt.figure(figsize=(15, 15))
for i in range(4):
    plt.subplot(1, 4, i+1)

    # Pick a random image from the 'cloudy' list
    x = random.randint(0, len(cloudy) - 1)
    img_path = os.path.join(cloudy_folder, cloudy[x])

    # Read and display the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('Cloudy')
    else:
        print(f"Warning: Could not load image {img_path}")

    plt.axis('off')

plt.tight_layout()
plt.show()


#rain images
# Define the path to the 'rain' images in Google Drive
rain_folder = '/content/gdrive/My Drive/dataset2/rain'

# Display random images from the 'rain' folder
plt.figure(figsize=(15, 15))
for i in range(4):
    plt.subplot(1, 4, i+1)

    # Pick a random image from the 'rain' list
    x = random.randint(0, len(rain) - 1)
    img_path = os.path.join(rain_folder, rain[x])

    # Read and display the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('Rain')
    else:
        print(f"Warning: Could not load image {img_path}")

    plt.axis('off')

plt.tight_layout()
plt.show()


#sunrise images
# Define the path to the 'sunrise' images in Google Drive
sunrise_folder = '/content/gdrive/My Drive/dataset2/sunrise'

# Display random images from the 'sunrise' folder
plt.figure(figsize=(15, 15))
for i in range(4):
    plt.subplot(1, 4, i+1)

    # Pick a random image from the 'sunrise' list
    x = random.randint(0, len(sunrise) - 1)
    img_path = os.path.join(sunrise_folder, sunrise[x])

    # Read and display the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('Sunrise')
    else:
        print(f"Warning: Could not load image {img_path}")

    plt.axis('off')

plt.tight_layout()
plt.show()


#sunny images
# Define the path to the 'shine' images in Google Drive
shine_folder = '/content/gdrive/My Drive/dataset2/shine'

# Display random images from the 'shine' folder
plt.figure(figsize=(15, 15))
for i in range(4):
    plt.subplot(1, 4, i+1)

    # Pick a random image from the 'sunshine' list
    x = random.randint(0, len(sunshine) - 1)
    img_path = os.path.join(shine_folder, sunshine[x])

    # Read and display the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('Sunshine')
    else:
        print(f"Warning: Could not load image {img_path}")

    plt.axis('off')

plt.tight_layout()
plt.show()


#dividing dataset into 3 parts
# Define the source and destination paths
source_path = '/content/gdrive/My Drive/dataset2'
output_path = '/content/gdrive/My Drive/datasetOutput/'

# Categories
categories = ['cloudy', 'rain', 'sunrise', 'shine']

# Split ratio: 80% train, 10% validation, 10% test
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure that the output path exists
os.makedirs(output_path, exist_ok=True)

# Loop through each category to split data
for category in categories:
    category_path = os.path.join(source_path, category)
    images = os.listdir(category_path)

    # Shuffle the images to randomize the split
    random.shuffle(images)

    # Calculate split sizes
    total_images = len(images)
    train_size = int(train_ratio * total_images)
    val_size = int(val_ratio * total_images)
    test_size = total_images - train_size - val_size

    # Create category-specific directories for train, val, and test
    category_train = os.path.join(output_path, 'train', category)
    category_val = os.path.join(output_path, 'val', category)
    category_test = os.path.join(output_path, 'test', category)

    os.makedirs(category_train, exist_ok=True)
    os.makedirs(category_val, exist_ok=True)
    os.makedirs(category_test, exist_ok=True)

    # Move images into respective directories
    for i, img in enumerate(images):
        img_path = os.path.join(category_path, img)
        if i < train_size:
            shutil.copy(img_path, category_train)
        elif i < train_size + val_size:
            shutil.copy(img_path, category_val)
        else:
            shutil.copy(img_path, category_test)

print("Dataset split completed successfully.")


# Number of files in train, val, and test directories
# Correct the directory paths
train_dir = os.path.join(output_path, 'train')
val_dir = os.path.join(output_path, 'val')
test_dir = os.path.join(output_path, 'test')

# Number of files in train, val, and test directories
total_train = 0
for root, dirs, files in os.walk(train_dir):
    total_train += len(files)
print('Train set size: ', total_train)

total_val = 0
for root, dirs, files in os.walk(val_dir):
    total_val += len(files)
print('Validation set size: ', total_val)

total_test = 0
for root, dirs, files in os.walk(test_dir):
    total_test += len(files)
print('Test set size: ', total_test)


!pip install tensorflow

## setting train and validatin path
train_data_path = '/content/gdrive/MyDrive/datasetOutput/train'
validation_data_path = '/content/gdrive/MyDrive/datasetOutput/val'

#### Defining all parameters for CNN
## Setting Image height and width to 150 each since all images are of varying size
img_width=150
img_height =150
### CNN architecture parameters
batch_size = 32
samples_per_epoch = 1000
validation_steps = 32
no_filters1 = 32
no_filters2 = 64
no_filters3=128
conv1_size = 3
conv2_size = 3
conv3_size=5
pool_size = 2
## no of classes according to dataset 4 in our case
classes_num = 4
## Defining initial learning rate
lr = 0.001
epochs=200

if not os.path.exists(train_data_path):
    print(f"Train path '{train_data_path}' not found.")
if not os.path.exists(validation_data_path):
    print(f"Validation path '{validation_data_path}' not found.")

train_datagen = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest",rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# #CNN
# #### Defining all parameters for CNN
# ## Setting Image height and width to 150 each since all images are of varying size
# img_width=150
# img_height =150
# ### CNN architecture parameters
# batch_size = 32
# samples_per_epoch = 1000
# validation_steps = 32
# no_filters1 = 32
# no_filters2 = 64
# no_filters3=128
# conv1_size = 3
# conv2_size = 3
# conv3_size=5
# pool_size = 2
# ## no of classes according to dataset 4 in our case
# classes_num = 4
# ## Defining initial learning rate
# lr = 0.001
# epochs=200

# if not os.path.exists(train_data_path):
#     print(f"Train path '{train_data_path}' not found.")
# if not os.path.exists(validation_data_path):
#     print(f"Validation path '{validation_data_path}' not found.")

# train_datagen = ImageDataGenerator(
# 		rotation_range=20,
# 		zoom_range=0.15,
# 		width_shift_range=0.2,
# 		height_shift_range=0.2,
# 		shear_range=0.15,
# 		horizontal_flip=True,
# 		fill_mode="nearest",rescale=1. / 255)
# test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_path,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_path,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')

# #Defining CNN architecture
# model = Sequential()
# # Corrected input shape definition
# model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same", input_shape=(img_width, img_height, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))

# # Remove `input_shape` from the following layers
# model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))

# model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Conv2D(filters=no_filters3, kernel_size=(5, 5), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(GlobalAveragePooling2D())
# model.add(Dense(1024))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))
# model.add(Dense(128))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))
# model.add(Dense(classes_num, activation='softmax'))

# # Print the summary of the model
# model.summary()

# # CNN - LSTM

# # Define CNN architecture (from your original code)
# model = Sequential()
# # Add CNN layers as you already have
# model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same", input_shape=(img_width, img_height, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))

# model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))

# model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Conv2D(filters=no_filters3, kernel_size=(5, 5), padding="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# # Flatten the CNN output to feed it to LSTM
# model.add(GlobalAveragePooling2D())

# # Reshape the CNN output into a sequence for LSTM
# model.add(Reshape((-1, 128)))  # Reshape it to be of shape (time_steps, features) for LSTM (can experiment with different time_steps)

# # Add LSTM layer
# model.add(LSTM(64, return_sequences=False))  # You can adjust the LSTM units as needed

# # Fully connected layers after LSTM
# model.add(Dense(1024))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))

# model.add(Dense(512))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))

# model.add(Dense(128))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))

# # Output layer
# model.add(Dense(classes_num, activation='softmax'))

# # Print the summary of the model
# model.summary()




# Define a simple Transformer block function
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-Head Attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    # Add & Layer Normalization
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization()(out1)

    # Feed-Forward Network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)

    # Add & Layer Normalization
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization()(out2)

    return out2

# Build the CNN part of the model
model = Sequential()
model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same", input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

model.add(Conv2D(filters=no_filters1, kernel_size=(3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

model.add(Conv2D(filters=no_filters2, kernel_size=(3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(filters=no_filters3, kernel_size=(5, 5), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# Flatten and reshape the output for the Transformer
model.add(GlobalAveragePooling2D())
model.add(Reshape((-1, 128)))  # Reshape as needed (e.g., sequence length, feature size)

# Define the input for the transformer model
input_shape = model.output_shape[1:]  # Get the shape of the input for the transformer
inputs = Input(shape=input_shape)

# Apply the Transformer block
transformer_output = transformer_block(inputs, num_heads=4, ff_dim=128)

# Create the Model object that includes the Transformer layer
transformer_model = Model(inputs=inputs, outputs=transformer_output)

# Integrate the Transformer model into the main model
model.add(transformer_model)  # Embed the Transformer model as a layer

# Fully connected layers after Transformer
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Flatten())  # Flatten the spatial dimensions

# Output layer
model.add(Dense(4, activation='softmax'))

# Print the summary of the model
model.summary()


## Defining Target directory for saving Models and weights
target_dir = './models'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
file_path = "models/best_weight{epoch:03d}.weights.h5"
chkpt = callbacks.ModelCheckpoint(
    file_path,
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto'
)

lr_red = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.1,
    patience=5,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=1e-30
)

cbks = [tb_cb, chkpt, lr_red]

# Training the model using fit
# Compile the model
model.compile(
    optimizer='adam',  # You can choose other optimizers like 'sgd', 'rmsprop', etc.
    loss='categorical_crossentropy',  # Adjust the loss function as needed
    metrics=['accuracy']  # You can add more metrics if necessary
)

# Training the model using fit
history = model.fit(
    train_generator,
    steps_per_epoch=130,
    epochs=200,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=32
)


#saving model
model.save('models/weather_model.keras')

#testing performance
from keras.models import load_model
from keras.preprocessing import image

# MOVING THE DATA FROM FOLDERS FOR TESTING PURPOSE
test_dir = '/content/gdrive/MyDrive/datasetOutput/test'
subfolders = ['rain', 'shine', 'cloudy', 'sunrise']

# Iterate through each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(test_dir, subfolder)

    if os.path.exists(subfolder_path):
        # List all files in the subfolder
        for img_file in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_file)

            # Check if the file is an image by its extension
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Move the image to the main test directory
                shutil.move(img_path, test_dir)
                print(f'Moved {img_file} from {subfolder} to {test_dir}')

        # Optionally remove the now-empty subfolder
        os.rmdir(subfolder_path)

print('All images have been moved to the main test directory.')


#testing
class_dict={ 0:'cloudy', 1:'rain', 2:'sunrise',3 :'shine'}
sunny1=os.listdir('/content/gdrive/MyDrive/datasetOutput/test')
#print(sunny1)
for img in sunny1:
    img_path = '/content/gdrive/MyDrive/datasetOutput/test/' + img
    if not os.path.exists(img_path):
        print(f"Image path not found: {img_path}")
        continue

    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        plt.imshow(img_tensor / 255)
        plt.show()

        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        pred = model.predict(img_tensor)
        print('Predicted class probability for image: ', pred)

        classes = np.argmax(pred)
        print('Predicted Class :', class_dict[classes])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")



import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Conv2DTranspose, Input
import os

# Allowing GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limiting GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def build_style_predict_model():
    input_layer = Input(shape=(256, 256, 3), name='style_image')
    x = Conv2D(32, 3, strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    for _ in range(5):
        x = Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    output_layer = Conv2D(3, 1, strides=1, padding='same')(x)
    model = Model(inputs=input_layer, outputs=output_layer, name='style_predict_model')
    return model

def build_style_transform_model():
    input_layer = Input(shape=(384, 384, 3), name='content_image')
    x = Conv2D(32, 3, strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    for _ in range(5):
        x = Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    output_layer = Conv2DTranspose(3, 9, strides=1, padding='same')(x)
    model = Model(inputs=input_layer, outputs=output_layer, name='style_transform_model')
    return model

def gram_matrix(x):
    batch, height, width, channels = tf.shape(x)
    features = tf.reshape(x, (batch, height * width, channels))
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(height * width * channels, tf.float32)

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = tf.shape(style)[-1]
    size = tf.cast(tf.reduce_prod(tf.shape(style)[:-1]), tf.float32)
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (tf.cast(channels, tf.float32) ** 2) * (size ** 2))

def content_loss(content, combination):
    return tf.reduce_mean(tf.square(content - combination))

def total_variation_loss(y_pred):
    a = tf.square(y_pred[:, :-1, :-1, :] - y_pred[:, 1:, :-1, :])
    b = tf.square(y_pred[:, :-1, :-1, :] - y_pred[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 2))

@tf.function
def total_loss(y_true, y_pred, content_features, style_features):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute style loss
    style_losses = [style_loss(style_feature, y_pred) for style_feature in style_features]
    style_loss_value = tf.reduce_sum(style_losses)

    # Compute content loss
    content_losses = [content_loss(content_feature, y_pred) for content_feature in content_features]
    content_loss_value = tf.reduce_sum(content_losses)
    
    tv_loss = total_variation_loss(y_pred)
    return mse_loss + style_weight * style_loss_value + content_weight * content_loss_value + total_variation_weight * tv_loss

# Hyperparameters
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Loading Style Images for styles to be transferred
style_predict_model = build_style_predict_model()
style_folder = "Style_Images"
style_images = []
style_files = os.listdir(style_folder)
for filename in style_files:
    if filename.endswith(".jpg"):
        style_path = os.path.join(style_folder, filename)
        style_image = tf.keras.preprocessing.image.load_img(style_path, target_size=(256, 256))
        style_array = tf.keras.preprocessing.image.img_to_array(style_image)
        style_array = tf.expand_dims(style_array, axis=0)
        style_images.append(style_array)

style_features = [style_predict_model.predict(style_array) for style_array in style_images]

# Loading content images and computing content features
content_folder = "resized_content_images"
content_images = []
content_files = os.listdir(content_folder)
for filename in content_files:
    if filename.endswith(".jpg"):
        content_path = os.path.join(content_folder, filename)
        content_image = tf.keras.preprocessing.image.load_img(content_path, target_size=(384, 384))
        content_array = tf.keras.preprocessing.image.img_to_array(content_image)
        content_array = tf.expand_dims(content_array, axis=0)
        content_images.append(content_array)

content_features = [style_predict_model.predict(content_array) for content_array in content_images]

# Creating and compiling the models
style_transform_model = build_style_transform_model()
style_transform_model.compile(optimizer='adam', loss=lambda y_true, y_pred: total_loss(y_true, y_pred, content_features, style_features))

# Training style_transform_model
total_epochs = 20
batch_size = 4
for epoch in range(total_epochs):
    print(f"Epoch {epoch+1}/{total_epochs}:")
    for i in range(0, len(content_images), batch_size):
        batch_content_images = tf.concat(content_images[i:i+batch_size], axis=0)
        batch_content_features_y = batch_content_images
        print("Training style_transform_model...")
        history = style_transform_model.fit(x=batch_content_images, y=batch_content_features_y, epochs=1, batch_size=batch_size, verbose=2)
        print("Loss:", history.history['loss'])

# Converting the models to TensorFlow Lite format
converter_predict = tf.lite.TFLiteConverter.from_keras_model(style_predict_model)
tflite_model_predict = converter_predict.convert()

converter_transform = tf.lite.TFLiteConverter.from_keras_model(style_transform_model)
tflite_model_transform = converter_transform.convert()

# Saving the TensorFlow Lite models to .tflite files
with open("style_predict.tflite", "wb") as f:
    f.write(tflite_model_predict)

with open("style_transform.tflite", "wb") as f:
    f.write(tflite_model_transform)

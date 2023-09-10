import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_cv
import cv2 as cv
import os

def rescaleFrame(frame,scale ):
    # this method works for: images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

num_skipped = 0
for folder_name in ("cat", "dog"):
    folder_path = os.path.join("imgs", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


image_size = (180, 180)
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "imgs",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


vis_ds = train_ds.take(1).unbatch()

vis_ds = vis_ds.take(8)


def get_images(image, _):
    return image


vis_ds = vis_ds.map(get_images)

vis_ds = vis_ds.apply(tf.data.experimental.dense_to_ragged_batch(8))

keras_cv.visualization.plot_image_gallery(
    next(iter(vis_ds.take(1))),
    value_range=(0, 255),
    scale=3,
    rows=4,
    cols=2,
)

data_augmentation = keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(
            value_range=(0, 255),
            augmentations_per_image=2,
            magnitude=0.5,
            magnitude_stddev=0.15,
        ),
    ]
)



vis_ds = vis_ds.map(data_augmentation)

keras_cv.visualization.plot_image_gallery(
    next(iter(vis_ds.take(1))),
    value_range=(0, 255),
    scale=3,
    rows=4,
    cols=2,
)


# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)


# Train the model


epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# still image

img = keras.utils.load_img("test/test.jpg", target_size=image_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


# live video
capture = cv.VideoCapture(0)

while True:
    isTrue,img = capture.read()
    cv.imshow("img",img)
    img = rescaleFrame(img, 0.40)
    
    img = img[0:180, 0:180]
    cv.imshow('Video resized',img)
    
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = float(predictions[0])
    cv.putText(img, "This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.", (100,100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
capture.release()
cv.destroyAllWindows()

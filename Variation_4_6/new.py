import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from SwinTransformer.models.swin_transformer import SwinTransformer 

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data paths
train_path = "train"
val_path = "dev"
test_path = "test"

# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation and test sets
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Load the Swin Transformer model
config = SwinConfig.from_pretrained('microsoft/swin-transformer-base')
base_model = SwinTransformer.from_pretrained('microsoft/swin-transformer-base', config=config)

# Add your own layers on top of the Swin Transformer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,  # You can adjust the number of epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Save the model
model.save('swin_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = np.round(predictions)

# Confusion matrix
conf_matrix = confusion_matrix(test_generator.classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(test_generator.classes, predicted_classes)
print("Classification Report:")
print(class_report)

# ROC curve
fpr, tpr, thresholds = roc_curve(test_generator.classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save predictions to CSV
filenames = test_generator.filenames
results = pd.DataFrame({'Filename': filenames, 'Predictions': predictions[:, 0]})
results.to_csv('swin_predictions.csv', index=False)

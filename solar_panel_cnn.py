import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf


#Create a variable for the metadata path
JSON_File_Path = "Raptor/module_metadata.json"

#save metadata as a dictionary
with open(JSON_File_Path, 'r') as f:
    module_metadata = json.load(f)

#check that the JSON file was loaded successfully
if isinstance(module_metadata, dict):
    print("JSON file loaded successfully.")
else:
    print("Error loading JSON file.")

#create empty lists for the np arrays to store data
image_ids = []
module_images = []
binary_classifications = []

#load metadata into the lists
for item in module_metadata:

    #add item_id to array
    image_ids.append(item)

    #add image to array
    image_path = "Raptor/" + module_metadata[item]["image_filepath"]
    image = cv2.imread(image_path)

    #check that the image has loaded
    if not isinstance(image, np.ndarray):
        print("Error loading image.")
    module_images.append(image)


    # change the classifications to 0 = Healthy and 1 = Faulty
    if module_metadata[item]['anomaly_class'] == 'No-Anomaly':
        binary_classifications.append(0)
    else:
        binary_classifications.append(1)

#convert lists to numpy arrays (W3Schools, n.d.)
image_ids = np.array(image_ids)
module_images = np.array(module_images)
binary_classifications = np.array(binary_classifications)

#check data accuracy
for i, item in enumerate(module_metadata):
    if item != image_ids[i]:
        print("item_id error")
    if module_metadata[item]['anomaly_class'] == "No-Anomaly":
        if binary_classifications[i] != 0:
            print("classification error")
    else:
        if binary_classifications[i] != 1:
            print("classification error")
    image_path = 'Raptor/' + module_metadata[item]["image_filepath"]
    image = cv2.imread(image_path)

    image_equal_check = np.array_equal(image, module_images[i])
    if not image_equal_check:
        print("image error")

#Scale the images (TensorFlow, 2024)
module_images = module_images / 255.0

#split the data into training and validation sets (scikit-learn developers, n.d.)
train_ids, val_ids, train_classes, val_classes, train_images, val_images \
    = train_test_split(image_ids, binary_classifications, module_images, test_size=0.3, random_state=36)

#split the validation data into validation and the test set for the user
val_ids, test_ids, val_classes, test_classes, val_images, test_images \
    = train_test_split(val_ids, val_classes, val_images, test_size=200, random_state=36)

#show the sample sizes
print(len(train_ids), 'training samples.')
print(len(val_ids), 'validation samples.')
print(len(test_ids), 'user testing samples.')

#save the data to a npz file for application use (kadambala, 2025)
np.savez('test_data.npz', test_ids, test_classes, test_images)

print(train_images.shape)

#The following lines 96-134 are modeled after the CNN tutorial by TensorFlow (TensorFlow, 2024) with modifications
#made to fit the image data from Raptor Maps Infrared Solar Modules (RaptorMaps, 2020)
#build the convolutional neural network
cnn_model = models.Sequential()
#first layer of convolution
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 24, 3)))
cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#second layer of convolution
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#final layer
cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#flatten the output from the convolutional layers
cnn_model.add(layers.Flatten())
#fully connected layer
cnn_model.add(layers.Dense(128, activation='relu'))
#binary dense output layer (Ali, 2025)
cnn_model.add(layers.Dense(1, activation='sigmoid'))
#compile the model (Brownlee, 2022)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
history = cnn_model.fit(train_images, train_classes, epochs=10, validation_data=(val_images, val_classes))

#save the model to a .keras file
cnn_model.save('solar_panel_cnn.keras')

#get some statistics on the model
with open('solar_panel_cnn_summary.txt', 'w', encoding='utf-8') as f:
    cnn_model.summary(print_fn=f.write)
cnn_model.summary()

#plot an accuracy graph over the epochs comparing accuracy against validation accuracy
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Accuracy History')
plt.savefig('accuracy_history.png')

#generate a confusion matrix lines 137-148 (Sharma, 2025)
predictions = cnn_model.predict(val_images)
#convert predictions to 0 or 1
predictions = (predictions > 0.5).astype(int)

cnn_model_cm = confusion_matrix(val_classes, predictions, labels=[0, 1])

display_cm = ConfusionMatrixDisplay(confusion_matrix=cnn_model_cm, display_labels=['0(neg)\nHealthy', '1(pos)\nFaulty'])
display_cm.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

#gather accuracy, recall, precision, f1 scores and confusion matrix numbers to a txt file (Singh, 2022)
with open('cnn_model_metrics.txt', 'w', encoding='utf-8') as f:
    cnn_model_accuracy = accuracy_score(val_classes, predictions)
    cnn_model_recall = recall_score(val_classes, predictions)
    cnn_model_precision = precision_score(val_classes, predictions)
    cnn_model_f1 = f1_score(val_classes, predictions)
    print(f"CNN Model Accuracy: {cnn_model_accuracy:.2f}", file=f)
    print(f"CNN Model Recall: {cnn_model_recall:.2f}", file=f)
    print(f"CNN Model Precision: {cnn_model_precision:.2f}", file=f)
    print(f"CNN Model F1 Score: {cnn_model_f1:.2f}", file=f)
    print(f"True negative: {cnn_model_cm[0][0]}", file=f)
    print(f"False positive: {cnn_model_cm[0][1]}", file=f)
    print(f"False negative: {cnn_model_cm[1][0]}", file=f)
    print(f"True positive: {cnn_model_cm[1][1]}", file=f)







#C964-Task 2: Design and Development Capstone
#Name: Jacob Taylor
#Student ID: 007025130
#Email: jta1001@wgu.edu
#Date: 12/22/2022

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

#For the streamlite my main source of information was gathered from kolheshivam17's (2025) geeks for geeks article
#streamlit title
st.title('Solar Panel Fault Detection')

#load the cnn_model
loaded_cnn_model = tf.keras.models.load_model('solar_panel_cnn.keras')

#load the test data
test_data = np.load('test_data.npz')

#load the np arrays from test_data
test_ids = test_data['arr_0']
test_classes = test_data['arr_1']
test_images = test_data['arr_2']

#create/show 25 sample images from the test data set (TensorFlow, 2024)
plt.figure(figsize=(14,14))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    #convert 0 or 1 to display a more readable label
    test_class = test_classes[i]
    if test_class == 1:
        test_class = 'Faulty Panel'
    else:
        test_class = 'Healthy Panel'
    plt.xlabel(test_class + '\n' + str(test_ids[i]))
st.image('data_reclassification.png', caption='Data reclassification from 12 to 2 classifications')
plt.suptitle('Sample Panel Images', fontsize=40)
#save the image of sample panels
plt.savefig('sample_solar_panel_images.png')
#show the image of sample panels
st.image('sample_solar_panel_images.png', caption='A sample of the first 25 solar panel images from the user testing dataset.')
#show the confusion matrix
st.image('confusion_matrix.png', caption='The confusion matrix for the binary classifications.\nNegative = Healthy Panel\nPositive = Faulty Panel')
#show the accuracy history graph
st.image('accuracy_history.png', caption='The accuracy of the training dataset against the validation dataset over the 10 epochs.')

#get the user input of an int in the range of the sample size
user_input = st.text_input('Enter a number between 1 and 200 to test the model on one of the test samples:')

#confirm the user input is a digit and if not throw an error message
if user_input.isdigit():
    #convert the user input into an int
    user_input = int(user_input)

    #check that the user input is within the valid range of the array
    if 1 <= user_input <= 200:
        print('You submitted: ', user_input, '')
        #grab the image data
        image_data = test_images[user_input-1]
        #convert the image before normalization for display
        image_data_display = (image_data * 255.0).astype(np.uint8)
        #convert the image to RGB
        image_rgb = cv2.cvtColor(image_data_display, cv2.COLOR_BGR2RGB)
        #convert numpy array to PIL image
        test_image = Image.fromarray(image_rgb)
        #convert test_class to string
        test_class = test_classes[user_input-1]
        if test_class == 1:
            test_class = 'Faulty Panel'
        else:
            test_class = 'Healthy Panel'
        #show the image
        st.image(test_image, caption= test_class + '\n' + str(test_ids[user_input-1]))

        #run the image through the cnn model
        test_image_batch = np.expand_dims(image_data, axis=0)
        predictions = loaded_cnn_model.predict(test_image_batch)
        #ensure that the predictions are binary
        predictions = (predictions > 0.5).astype(int)

        #show what to model predicts
        if predictions == 0:
            print('The model predicts that it is a Healthy Panel')
            st.success('The model predicts that it is a Healthy Panel')
        else:
            print('The model predicts that it is a Faulty Panel')
            st.success('The model predicts that it is a Faulty Panel')

        #show if the model predicted correctly or incorrectly
        if predictions == test_classes[user_input-1]:
            print('The model correctly predicted the panel\'s status.')
            st.success('The model correctly predicted the panel\'s status.')
        else:
            print('The model incorrectly predicted the panel\'s status.')
            st.error('The model incorrectly predicted the panel\'s status.')

    #out-of-range message
    else:
        st.text('Please enter a valid number between 1 and 200.')
#incorrect input message
else:
    st.text('Please enter a valid number between 1 and 200.')

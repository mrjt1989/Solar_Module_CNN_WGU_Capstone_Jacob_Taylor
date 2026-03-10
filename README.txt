Web-based access:

1.	Navigate in your preferred browser to: https://solarmodulecnnjacobtaylorcapstone.streamlit.app/

2.	It may take a moment to load.

3.	When the page loads, the title should read “Solar Panel Fault Detection.”

4.	As you scroll down the page, there will be four visualizations that represent this project.

5.	At the bottom of the page, under the “Accuracy History” graph, there will be the following instructions:
a.	Enter a number between 1 and 200 to test the model on one of the test samples:

6.	Directly below that, there is an area to input any number between 1 and 200 to grab one of the sample images to test the CNN model.
7.	After the number is entered, an IR image of the panel will be displayed with its image ID and classification

8.	The model will then predict whether the panel is faulty or healthy and tell you if it predicted accurately

Locally Run
1.	Download the zip file of the project: C964_Solar_Module_CNN_Jacob_Taylor

2.	Extract the file


3.	Install Python 3.12.10 https://www.python.org/downloads/ (it will be under the “Looking for a specific release” section

4.	Install PyCharm 2025.3.1


5.	Open the unzipped file: C964_Solar_Module_CNN_Jacob_Taylor as a project

6.	Open the terminal in PyCharm, which can be found in the bottom left sidebar (or the Alt + F12 hotkey)


7.	Run the following command in the terminal to install all the packages and requirements:
pip install -r requirements.txt

8.	After everything has been installed, in the terminal run:
streamlit run main.py

9.	This will open this page in your default web browser: http://localhost:8501/

10.	The app will load, and please follow the instructions for web-based access above, starting at step 3.


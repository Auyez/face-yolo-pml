In order to run our project the following packages for python 3 are required:
-numpy
-cv2 (opencv-python)
-pillow
-plaidml

PlaidML installation:

Original instructions: https://github.com/plaidml/plaidml/blob/master/docs/installing.md

You can install it simply by calling
	sudo -H pip install -U plaidml-keras
and then by running setup
	plaidml-setup
In case of errors you can use link provided above.

Project can run without PlaidML
In this case you will need Keras installed, and you should comment the following lines in predict.py or webcam.py
or in any python file that you want to run:
import plaidml.keras
plaidml.keras.install_backend()

These two lines must be commented out in order to not use plaidml. Then Keras will use tensorflow or any other available backend.

Running network

To test network on an image you will need to run following command:
python src/predict.py PATH_TO_YOUR_IMAGE_FILE
To test network on webcam you will need to run following command:
python src/webcam.py


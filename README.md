# 📖 Lip Reading Application 👄🎥

This project is a lip reading application that uses computer vision and deep learning to recognize spoken words from lip movements in short video clips. It consists of two main components:

- **`lip_extractor.py`**: Extracts lip regions from video frames using MediaPipe's Face Mesh and creates a grid image of detected lips.
- **`app.py`**: A Streamlit web interface that allows users to upload or record a short video, extract the lips, and get a predicted spoken word using a pre-trained model.

---

🎯 Supported Words
The model is trained to recognize the following words/phrases (Dataset From Kaggle):

Stop navigation

Excuse me

I am sorry

Thank you

Good bye

I love this game

Nice to meet you

You are welcome

How are you?

Have a good time

Begin

Choose

Connection

Navigation

Next

Previous

Start

Stop

Hello

Web

🧠 How It Works
Lip Extraction (lip_extractor.py)
Uses MediaPipe Face Mesh to locate facial landmarks.

Extracts the lip region and resizes it to a uniform size.

Assembles a fixed number (49 by default) of frames into a single grid image.

Prediction (app.py)
Loads a trained Keras model (.h5) that was trained on lip images corresponding to specific words.

Preprocesses the extracted lip grid image and feeds it into the model.

Displays the predicted word on the UI.

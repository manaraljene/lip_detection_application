import streamlit as st
import os
import cv2
import tensorflow as tf
from lip_extractor import process_video
from tensorflow.keras.models import load_model
import numpy as np

# Charger la liste des mots cibles
WORDS_LIST = [
    "Stop navigation .", "Excuse me .", "I am sorry .", "Thank you .", "Good bye .", "I love this game .",
    "Nice to meet you .", "You are welcome", "How are you ?", "Have a good time .",
    "Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"
]

# Charger le modèle sauvegardé
model_path = "C:/Users/manar/Documents/pfa_app/app1/lip_reading_model.h5"
model = load_model(model_path)

# Titre de l'application
st.title("Lip Reading App 🎥👄")

# Diviser l'interface en deux colonnes
col1, col2 = st.columns([1, 3])  # La colonne gauche est plus étroite que la droite

# Ajouter une image dans la colonne de gauche
with col1:
    st.image("C:/Users/manar/Pictures/Screenshots/mots.png", caption="Les mots possibles", use_container_width=True)

# Contenu principal dans la colonne de droite
with col2:
    # Options pour l'utilisateur : charger une vidéo ou capturer en temps réel
    option = st.radio(
        "Choisissez une option :",
        ("Charger une vidéo existante", "Capturer une vidéo avec la caméra (2 secondes)")
    )

    # Charger une vidéo existante
    if option == "Charger une vidéo existante":
        # Téléchargement de la vidéo
        uploaded_video = st.file_uploader("Téléchargez une vidéo", type=["mp4", "avi", "mov"])
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("extracted", exist_ok=True)

        if uploaded_video is not None:
            video_path = os.path.join("uploads", uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.success("✅ Vidéo téléchargée!")

            if st.button("Extraire les lèvres et prédire"):
                output_path = process_video(video_path, "extracted", grid_size=(7, 7))

                if output_path:
                    st.image(output_path, caption="Grille des lèvres extraites")
                    st.success("✅ Région des lèvres extraite!")

                    # Charger et prétraiter l'image de grille pour la prédiction
                    lip_grid = tf.keras.preprocessing.image.load_img(output_path, target_size=(224, 224))
                    lip_grid = tf.keras.preprocessing.image.img_to_array(lip_grid)
                    lip_grid = np.expand_dims(lip_grid, axis=0)
                    lip_grid = lip_grid / 255.0

                    # Effectuer une prédiction
                    predictions = model.predict(lip_grid)
                    predicted_word_index = np.argmax(predictions)
                    predicted_word = WORDS_LIST[predicted_word_index]

                    st.success(f"Le mot prédit est : **{predicted_word}** 👄")
                else:
                    st.error("❌ Aucune lèvre détectée.")

    # Capturer une vidéo de 2 secondes avec la webcam
    elif option == "Capturer une vidéo avec la caméra (2 secondes)":
        if st.button("Démarrer la capture vidéo"):
            # Démarrer la capture vidéo
            cap = cv2.VideoCapture(0)  # Webcam par défaut
            os.makedirs("real_time_videos", exist_ok=True)
            video_path = "real_time_videos/real_time_video.mp4"

            # Définir le codec et le format vidéo
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))  # FPS = 20, résolution = 640x480

            st.write("⏳ Enregistrement pendant 2 secondes...")
            start_time = cv2.getTickCount()
            duration_seconds = 2  # 2 secondes

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)  # Écriture des frames dans la vidéo

                # Calcul du temps écoulé
                elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed_time >= duration_seconds:
                    break

            # Libérer les ressources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            st.success("✅ Vidéo capturée!")

            # Traiter la vidéo capturée
            st.write("🔄 Transformation en matrice et prédiction...")
            output_path = process_video(video_path, "extracted", grid_size=(7, 7))

            if output_path:
                st.image(output_path, caption="Grille des lèvres extraites")
                st.success("✅ Région des lèvres extraite!")

                # Prétraiter l'image de la grille
                lip_grid = tf.keras.preprocessing.image.load_img(output_path, target_size=(224, 224))
                lip_grid = tf.keras.preprocessing.image.img_to_array(lip_grid)
                lip_grid = np.expand_dims(lip_grid, axis=0)
                lip_grid = lip_grid / 255.0

                # Effectuer une prédiction
                predictions = model.predict(lip_grid)
                predicted_word_index = np.argmax(predictions)
                predicted_word = WORDS_LIST[predicted_word_index]

                st.success(f"Le mot prédit est : **{predicted_word}** 👄")
            else:
                st.error("❌ Aucune lèvre détectée.")
import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Entraînement VGG16 pour la détection du Cube Rouge\n",
                "Ce notebook charge le dataset généré par Webots, prépare les images, et entraîne un réseau VGG16 modifié pour prédire les coordonnées X et Y du cube en pixels."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import cv2\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.model_selection import train_test_split\n",
                "from tensorflow.keras.applications import VGG16\n",
                "from tensorflow.keras.models import Model, load_model\n",
                "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n",
                "from tensorflow.keras.optimizers import Adam\n",
                "\n",
                "print('Librairies chargées avec succès !')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Chargement des données (Images + CSV)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Le script cherche intelligemment où se trouve le dataset\n",
                "if os.path.exists('../dataset/labels.csv'):\n",
                "    dataset_dir = '../dataset'\n",
                "elif os.path.exists('./dataset/labels.csv'):\n",
                "    dataset_dir = './dataset'\n",
                "else:\n",
                "    # Au pire des cas, on force le chemin absolu direct\n",
                "    dataset_dir = r'reference_ur5_repo\\dataset'\n",
                "\n",
                "csv_path = os.path.join(dataset_dir, 'labels.csv')\n",
                "img_dir = os.path.join(dataset_dir, 'images')\n",
                "\n",
                "df = pd.read_csv(csv_path)\n",
                "print(f'Nombre total d\\'images trouvées : {len(df)}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_data = []\n",
                "Y_data = []\n",
                "\n",
                "for index, row in df.iterrows():\n",
                "    img_path = os.path.join(img_dir, row['filename'])\n",
                "    img = cv2.imread(img_path)\n",
                "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Keras s'attend à du RGB\n",
                "    # Normalisation des pixels entre 0 et 1 (facilite l'apprentissage)\n",
                "    img = img / 255.0\n",
                "    \n",
                "    X_data.append(img)\n",
                "    Y_data.append([row['x_pixel'], row['y_pixel']])\n",
                "\n",
                "X_data = np.array(X_data)\n",
                "Y_data = np.array(Y_data)\n",
                "\n",
                "print('Forme des images :', X_data.shape)\n",
                "print('Forme des labels :', Y_data.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Séparation Entraînement / Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)\n",
                "print(f'Images pour l\\'entraînement : {len(X_train)}')\n",
                "print(f'Images pour la validation : {len(X_val)}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Création OU Chargement du modèle VGG16"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_path = 'vgg16.h5'\n",
                "\n",
                "if os.path.exists(model_path):\n",
                "    print(f'Modèle existant trouvé ({model_path}) ! Chargement en cours pour gagner du temps...')\n",
                "    # On désactive la compilation à l'ouverture pour éviter le bug Keras 'mse' \n",
                "    model = load_model(model_path, compile=False)\n",
                "    # On recompile manuellement ensuite\n",
                "    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])\n",
                "    print('Modèle chargé avec succès !')\n",
                "    \n",
                "else:\n",
                "    print('Aucun modèle existant trouvé. Création d\\'un nouveau réseau...')\n",
                "    # On charge VGG16 SANS sa tête de classification (include_top=False)\n",
                "    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))\n",
                "\n",
                "    # On fige le modèle de base pour ne pas détruire ce qu'il a déjà appris sur les formes/lignes\n",
                "    for layer in base_model.layers:\n",
                "        layer.trainable = False\n",
                "\n",
                "    # On ajoute notre propre 'tête' de régression (pour prédire 2 chiffres : X et Y)\n",
                "    x = base_model.output\n",
                "    x = GlobalAveragePooling2D()(x)\n",
                "    x = Dense(128, activation='relu')(x)\n",
                "    x = Dense(64, activation='relu')(x)\n",
                "    predictions = Dense(2, activation='linear')(x) # Sortie linéaire pour des coordonnées\n",
                "\n",
                "    model = Model(inputs=base_model.input, outputs=predictions)\n",
                "\n",
                "    # Utilisation de MSE (Mean Squared Error) adapté pour prédire des coordonnées continues\n",
                "    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])\n",
                "    model.summary()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Entraînement de l'IA (Ignoré si déjà entraîné)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if not os.path.exists(model_path):\n",
                "    print('Début de l\\'entraînement...')\n",
                "    history = model.fit(\n",
                "        X_train, y_train,\n",
                "        validation_data=(X_val, y_val),\n",
                "        epochs=15, # Tu peux augmenter à 30 si ce n'est pas assez précis\n",
                "        batch_size=32\n",
                "    )\n",
                "    \n",
                "    model.save(model_path)\n",
                "    print(f'Modèle sauvegardé sous le nom {model_path} !')\n",
                "else:\n",
                "    print('Modèle déjà entraîné ! Étape ignorée.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Test Visuel"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# On prend une image au hasard dans le set de validation\n",
                "idx = np.random.randint(0, len(X_val))\n",
                "test_img = X_val[idx]\n",
                "true_y = y_val[idx]\n",
                "\n",
                "# L'IA fait sa prédiction (à l'échelle 512)\n",
                "pred = model.predict(np.expand_dims(test_img, axis=0))[0]\n",
                "\n",
                "# On divise par 2 juste pour l'affichage visuel sur la petite image 256x256\n",
                "true_y_vis = true_y / 2.0\n",
                "pred_vis = pred / 2.0\n",
                "\n",
                "plt.imshow(test_img)\n",
                "plt.scatter(true_y_vis[0], true_y_vis[1], c='green', marker='x', s=100, label='Vrai')\n",
                "plt.scatter(pred_vis[0], pred_vis[1], c='red', marker='o', s=50, label='Prédiction IA')\n",
                "plt.legend()\n",
                "plt.title(f'Prédiction : X={pred[0]:.1f}, Y={pred[1]:.1f} (Echelle 512)')\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), 'train_vgg16.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print('Notebook created successfully.')

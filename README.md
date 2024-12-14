# Deep Learning Project

This project focuses on developing a neural network-based music genre classification tool using a
dataset with five music genres. The dataset is analyzed through descriptive statistics and baseline
classification using baseline Machine Learning models. A neural network model is then designed and
optimized by experimenting with different hyperparameters, using heuristics like batch normalization,
dropout, and gradient clipping to improve performance. Predictions are made for the test set, and
results are exported. The model is extended by incorporating textual information, and the neural
network is retrained with a new dataset containing an additional genre, using transfer learning.

Keywords: music genre classification, neural networks, machine learning, transfer learning, textual data

## Project Structure


## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- XGBoost
- Pandas
- Matplotlib
- Other dependencies in `requirements.txt`

### Get the Kaggle data here

spotify_songs <- https://www.kaggle.com/code/jgabrielsb/spotify-songs-music-genre-predictor-part-i/notebook

### Data Dictionary

# `spotify_songs.csv`

|variable                 |class     |description |
|:---|:---|:-----------|
|track_id                 |character | Song unique ID|
|track_name               |character | Song Name|
|track_artist             |character | Song Artist|
|track_popularity         |double    | Song Popularity (0-100) where higher is better |
|track_album_id           |character | Album unique ID|
|track_album_name         |character | Song album name |
|track_album_release_date |character | Date when album released |
|playlist_name            |character | Name of playlist |
|playlist_id              |character | Playlist ID|
|playlist_genre           |character | Playlist genre |
|playlist_subgenre        |character | Playlist subgenre|
|danceability             |double    | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
|energy                   |double    | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
|key                      |double    | The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = Câ™¯/Dâ™­, 2 = D, and so on. If no key was detected, the value is -1. |
|loudness                 |double    | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.|
|mode                     |double    | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.|
|speechiness              |double    | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |
|acousticness             |double    | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.|
|instrumentalness         |double    | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
|liveness                 |double    | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
|valence                  |double    | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
|tempo                    |double    | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
|duration_ms              |double    | Duration of song in milliseconds |



## Project Organization


Here’s an overview of the folder and file structure in the project:

## Root Directory
- `.gitattributes` - Git configuration for file attributes
- `.gitignore` - Specifies files to be ignored by Git
- `config.py` - Configuration file for constants and settings
- `main.py` - Main script to run the project
- `README.md` - Project overview and instructions
- `requirements.txt` - Python dependencies for the project

## Data Directory
- `spotify_songs.csv` - Full dataset for Spotify song information
- `test.csv` - Test dataset for model evaluation
- `train.csv` - Training dataset for model development

## EDA (Exploratory Data Analysis)
- `exploratory_data_analysis.ipynb` - Notebook for data exploration and visualization

## Models Directory
- `model_local_and_numerical_and_textual__data.pth` - Neural network model checkpoint

### Subdirectories:
- **randomforestmodel/** - Folder for Random Forest model files
- **xgboostmodel/** - Folder for XGBoost model files
  - `xgboostmodel_model.joblib` - Trained XGBoost model

## Report Directory
- `case_study.pdf` - Case study document
- `report_ELADLOUNI_MOHAMMEDALI.pdf` - Final project report

## Results Directory
- `neural_network_hypermarameter_tuning_results._for_numerical_data.csv` - Results of neural network hyperparameter tuning

### Subdirectories:
- **genre_classification_predictions/** - Model predictions for genre classification
  - `ELADLOUNI_prediction_numerical_and_textual.csv`
  - `ELADLOUNI_prediction_numerical_data.csv`

- **neural_networks_losses/** - Loss graphs for neural network training
  - `loss_textual.png` - Loss graph for textual features
  - `numerical_and_textual_loss.png` - Loss graph for combined features
  - `training_loss_nn.png` - Training loss graph for neural networks
  - `transfer_learning_loss.png` - Transfer learning loss graph

- **xgboostmodel/** - Files for XGBoost model
  - `best_accuracy.json` - Best accuracy metrics for XGBoost model
  - `best_params.json` - Best hyperparameters for XGBoost model
  - `learning_curve_xgboostmodel.png` - Learning curve of XGBoost model

  ### Subdirectory:
  - **EDA_plots/** - Plots from EDA
    - `path_to_your_correlation_matrix_plot.png` - Correlation matrix visualization
    - `path_to_your_genre_distribution_plot.png` - Genre distribution plot

## Source Code (src) Directory
- `data_loader.py` - Module for loading and preprocessing data
- `models.py` - Defines machine learning and neural network models
- `training.py` - Handles training pipeline for models
- `utils.py` - Utility functions for common tasks
- `__init__.py` - Marks `src` as a package







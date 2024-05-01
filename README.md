# GDMS - Engine Detection
# S24-20 MDE Team
## Installing Dependencies
### External Software Needed
- Python 3.9 or greater
- FFmpeg
- Tkinter
### Python Libraries Needed
The libraries required for this project are included in `requirements.txt` inside the project root directory. Install them using the command
```
pip3 install -r requirements.txt
```
Note: If you are running this project on a Windows machine, you will need to uncomment the lines specified in the requirements file.
<br>
Once these dependencies have been installed, the project will be able to run without an internet connection.

## Training Setup
This project requires a specific folder structure for the audio training data. The top-level training data folder should contain a folder for each data class, and each of those folders should contain audio files corresponding to its class. In this project, the data classes are car brands. An example of the expected folder structure is shown below.
```
training_data_dir
|
|___BMW
|   |   128i.wav
|   |   M3.wav
|   |   ...
|
|___Ford
|   |   Explorer.wav
|   |   F150.wav
|   |   ...
|
|___Toyota
|   |   Corolla.wav
|   |   Camry.wav
|   |   ...
|
...
```

## Running the Training Script
First, the audio needs to be processed and turned into a form that is suitable for input to a neural network. This project uses a single Python script for this at `src/scripts/augment_segment_mfcc_extract_master.py` to perform the entire audio processing task, for ease of use. This includes applying data augmentation, segmentation, and MFCC extraction. The required arguments are:
```
--input_dirs <training audio directory>
--output_dir <MFCC output directory>
```

To create an engine classifier model that's compatible with the full program, it needs to use 100ms audio segments with 13 MFCCs per sample. The following arguments will set these required parameters (13 MFCCs is the script's default).
```
python3 augment_segment_mfcc_extract_master.py --input_dirs <training audio directory> --output_dir <MFCC output directory> --segment_length 100
```

After the MFCC file has been created, the neural network is ready to be trained. This is done using the script at `src/scripts/train.py`. By default, this script reads MFCC vectors from `src/trained_models/MFCC_top_10_brands.npz`. The MFCC file location can be changed using the `--mfccdir` and `--mfcc_filename` arguments. Lastly, the script saves the fully trained model at `src/trained_models/engine_classify.keras`. The training script can be run as
```
python3 train.py --mfccdir <directory> --mfcc_filename <file.npz>
```
## Running the Full Program
The program needs certain files to be able to run properly. This includes ML model files and car brand image files. The required files are listed below.
```
src/trained_models/engine_classify_10_brands_final.keras
src/trained_models/RandomForest_pipeline.joblib
src/trained_models/label_to_category_top_mapping.pickle
all .png files in src/photos
```

Note: RandomForest_pipeline.joblib is too large to be included in the GitHub repo as is. To use it, you must extract it from `src/trained_models/RandomForest_pipeline.zip`.
<br>
To run the program, navigate to the `src` directory in a command prompt and run the following command:
```
python3 -m engine_detection
```
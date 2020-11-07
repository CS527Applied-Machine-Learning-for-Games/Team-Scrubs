For the first time, setting up the environment with the requirements.txt in PythonFiles:

Install the lastest version of Anaconda following the instruction on the official website.

```
cd PythonFiles
conda create -n csci527 python=3.7.4
conda activate csci527
conda install pip
pip install -r requirements.txt
```

To run the back end, `python main.py`

The first time you run main.py, it will take a while to pretrained the model. After the pre-training, the backend will be checking drawings generated from the frontend every 10 seconds. Whenever there are more than `play_batch_size` of drawings, the model will train a batch with the new drawings. If you stop running the program at anytime, it should pickup from based on checkpoints in the progress.

If you have ran this program before, and
1. you want to restart the game, or
2. meet an error continue previous game.

Go throught the following reset process:

1. Make sure you remove all generated files in
    1. `PythonFiles/model/*.h5`,
    2. `UnityProject/Assets/Resources/data/drawings`,
    3. `UnityProject/Assets/Resources/data/reports`,
    4. `UnityProject/Assets/Resources/data/predictions`

2. Also, change `pretrain_finished` and `current_model` parameters to 0 in `state.properties` file.

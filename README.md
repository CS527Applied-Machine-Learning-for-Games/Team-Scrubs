# Team-Scrubs

There are two files that are too big to be uploaded to GitHub. They are:
1) "mixlabel_z_12_12_19_model.h5" which is the pre-trained machine learning segmentation model
2) "Stevens_PBC_INS1e_25mMglu+Ex-4_5min_1083_16-17_pre_rec.nii.gz" which is a large 3D image file

Please download these files from the folder here: https://drive.google.com/drive/folders/1YCoVf4BxkDhh0I_YOZa6LczGR_N6g4an?usp=sharing and place them in "Team-Scrubs\PythonFiles\ML code\scripts\models\" and "Team-Scrubs\PythonFiles\ML code\data\Image_3D", respectively.

This project currently has two demos that are completely separate. 
1) Paint Brush Canvas image saving + server/client python communication 
2) machine learning segmentation prediction

For Demo 1:

Install Unity version 2020.1.5f1 and open the project folder "Team-Scrubs/Unity Project".

Load the scene "Assets/Scenes/Paint Canvas Scene". Run the scene and see if you can paint and save the image.
Image will be saved in the project folder, assets/createdImages.

Now, with the unity scene still running, open up your command line. Navigate to "Team-Scrubs/PythonFiles" and enter 'python server.py' you should see an exchange of "Hello" and "World" between the Unity Debug console and command line.


For Demo 2:

Open up your command line. Navigate to "Team-Scrubs/PythonFiles/ML code/scripts".

Install the dependencies "pip install -r requirements_test.txt"

Run "python unet_predict_JP_2Dfor3D_noEnsemble.py 0".

This will read in 512 image files from "Team-Scrubs\PythonFiles\ML code\data\Image_3D\Stevens_PBC_INS1e_25mMglu+Ex-4_5min_1083_16-17_pre_rec\z" and save 512 predictions to "Team-Scrubs\PythonFiles\ML code\scripts\predictions".

To view these images, download the NIH-funded viewer ITK-SNAP from http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3.

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Hosting;
using UnityEngine;
using UnityEngine.SceneManagement;
using SimpleFileBrowser;

public class MainMenu : MonoBehaviour
{
    void Start()
    {
        saveCurrentLocation();
    }

    private void saveCurrentLocation()
    {
        string path = "Assets/Resources/data/player_data.txt";
        int currentImg = PlayerPrefs.GetInt("imgNum", 1);
        using (StreamWriter writer = new StreamWriter(path))  
        {  
            writer.WriteLine(currentImg.ToString());
        }
    }
    
    public void PlayGame()
    {
        //SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
        PlayerPrefs.SetString("StartGame","go");
        SceneManager.LoadScene("Cutscene");
    }

    public void RestartGame()
    {
        PlayerPrefs.DeleteAll();
        PlayGame();
    }

    public void ULdrawing()
    {
        FileBrowser.ShowLoadDialog((paths) => 
            { 
                Debug.Log("Selected: " + paths[0]);
                string[] files = Directory.GetFiles(paths[0]);

                foreach(string file in files)
                {
                    File.Copy(file, Application.dataPath + "/Resources/data/user_drawings/" + Path.GetFileName(file));
                }
            },
                () => { Debug.Log( "Canceled" ); },
                true, false, null, "Select Folder", "Select" );

    }

    public void ULlabel()
    {

        FileBrowser.ShowLoadDialog((paths) =>
        {
            Debug.Log("Selected: " + paths[0]);
            string[] files = Directory.GetFiles(paths[0]);

            foreach (string file in files)
            {
                File.Copy(file, Application.dataPath + "/Resources/data/user_labels/" + Path.GetFileName(file));
            }
        },
                () => { Debug.Log("Canceled"); },
                true, false, null, "Select Folder", "Select");

    }

}

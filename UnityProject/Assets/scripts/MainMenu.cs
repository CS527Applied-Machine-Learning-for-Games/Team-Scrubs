using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    void Start()
    {
        //***Must run this line below when truly first time run the game***
        // TODO: Create a reset to start of game button and call PlayerPrefs.DeleteAll() and the rest of the code below.
        PlayerPrefs.DeleteAll();
        
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

    public void QuitGame()
    {
        Debug.Log("Quit!");
        Application.Quit();
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System.Globalization;
using System.Security.Cryptography;

public class CutsceneLoad : MonoBehaviour
{
    public GameObject letters;
    public static string para0 = "This is my opening cutscene.";
    public static string para1 = "During an ordinary quarantine days...\nA dumb-looking robot knocked your door and delivered a letter to you. You open the envelope…\nGood news! Your COVID testing result is negative. But says the robot, “my hospital is so short of staff and at the edge of breaking down. I really want to help but not smart enough. Can you train me to be smarter, so that I can go help doctors to defeat COVID!";
    public static string para2 = "This is my cutscene story 2.";
    public static string para3 = "This is my cutscene story 3.";
    private IList<string> myScenes = new List<string>() { para0, para1,para2,para3};


    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(WaitForCutscene());

    }

    private IEnumerator WaitForCutscene()
    {
        if (PlayerPrefs.HasKey("StartGame"))
        {
            PlayerPrefs.DeleteKey("StartGame");
            Text myText = letters.GetComponent<Text>();
            myText.text = myScenes[0];
        }
        else
        {
            var myNum = PlayerPrefs.GetInt("imgNum", 1) / 10;
            Text myText = letters.GetComponent<Text>();
            myText.text = myScenes[myNum];
        }
        
        yield return new WaitForSeconds(5);
        SceneManager.LoadScene("Paint Canvas Scene");
    }

    public void skipCutscene()
    {
        SceneManager.LoadScene("Paint Canvas Scene");
    }

}

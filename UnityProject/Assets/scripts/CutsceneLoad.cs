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
    public static string para1 = "This is my cutscene story 1.";
    public static string para2 = "This is my cutscene story 2.";
    public static string para3 = "This is my cutscene story 3.";
    private IList<string> myScenes = new List<string>() { para1,para2,para3};


    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(WaitForCutscene());

    }

    private IEnumerator WaitForCutscene()
    {
        var myNum = PlayerPrefs.GetInt("imgNum", 1) / 10;
        Text myText = letters.GetComponent<Text>();
        myText.text = myScenes[myNum - 1];

        yield return new WaitForSeconds(5);
        SceneManager.LoadScene("Paint Canvas Scene");
    }

    public void skipCutscene()
    {
        SceneManager.LoadScene("Paint Canvas Scene");
    }

}

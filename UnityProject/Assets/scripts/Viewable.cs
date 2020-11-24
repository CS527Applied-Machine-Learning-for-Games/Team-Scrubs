using System.Collections;
using UnityEngine;
using System;
using System.IO;

public class Viewable : MonoBehaviour
{
    public int imgNumber = 1;
    public string resourcePath;

    private Texture[] allTextures;
    private Texture[] predictedTextures;
    
    private Renderer _renderer;

    public Texture showHintTexture;
    

    // Start is called before the first frame update
    void Start()
    {
        _renderer = GetComponent<Renderer>();
        imgNumber = PlayerPrefs.GetInt("imgNum", 1);
        
        string imageDirectory = Application.streamingAssetsPath + resourcePath;

        /*
        allTextures = Resources.LoadAll<Texture>(Application.streamingAssetsPath + "/Resources/" + resourcePath);
        Array.Sort<Texture>(allTextures,
            delegate (Texture x, Texture y) { return String.Compare(x.name, y.name, StringComparison.Ordinal); });

        Array.Sort(allTextures, delegate (Texture x, Texture y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });
        */

        string[] files = Directory.GetFiles(imageDirectory, "*png");

        allTextures = new Texture2D[files.Length];

        Array.Sort<string>(files,
            delegate (string x, string y) { return String.Compare(x, y, StringComparison.Ordinal); });

        //Does this need to work?

        /*Array.Sort(files, delegate (string x, string y) {
            return Int16.Parse(x).CompareTo(Int16.Parse(y));
        });*/

        for(int i = 0; i<Math.Min(files.Length, 20); i++) // get the first 20 images
        {
            Texture2D tex = null;
            byte[] imageData;
            imageData = File.ReadAllBytes(files[i]);

            tex = new Texture2D(1, 1);
            tex.LoadImage(imageData);

            allTextures[i] = tex;
        }
        
        updatePredictedTextures();
        
        _renderer.material.mainTexture = showHintTexture;

    }

    // Update is called once per frame
    void Update()
    {
     
    }

    
    public void goToNextImage()
    {
        
        StartCoroutine(coNext());

    }

    private IEnumerator coNext()
    {
        yield return new WaitForEndOfFrame();

        imgNumber += 1;

        // Go To Next Image
        next(imgNumber);

    }

    public void next(int imgNum)
    {
        // swap material texture
        imgNumber = imgNum;
        _renderer.material.mainTexture = showHintTexture;
        //_renderer.material.mainTexture = allTextures[imgNumber - 1];
        
        updatePredictedTextures();
    }

    public void showHint()
    {
        _renderer.material.mainTexture = allTextures[imgNumber - 1];
    }

    private void updatePredictedTextures()
    {
        string predictedDirectory = Application.streamingAssetsPath + "/data/predictions";
        string[] predictedFiles = Directory.GetFiles(predictedDirectory, "*png");

        Array.Sort<string>(predictedFiles,
            delegate (string x, string y) { return String.Compare(x, y, StringComparison.Ordinal); });

        
        for(int i = 20; i<20+predictedFiles.Length; i++) // get predicted image 20 - most current
        {
            Texture2D tex = null;
            byte[] imageData;
            imageData = File.ReadAllBytes(predictedFiles[i-20]);

            tex = new Texture2D(1, 1);
            tex.LoadImage(imageData);

            allTextures[i] = tex;
        }
    }
    

}


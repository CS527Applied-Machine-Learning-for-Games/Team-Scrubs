
using System;
using System.Collections;
using System.IO;
using UnityEngine;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public RenderTexture RTexture1;
    public Viewable viewable; 
    
    public float BrushSize = 0.07f;
    public float upMultiplier = 0.00f;

    public int imgNumber = 1;
    public string resourcePath;

    private Texture2D[] allTextures;
    private Texture2D[] labelTextures;
    
    private Renderer _renderer;

    private int _isRunning = 1;
    private int _secFreqRefresh = 1;
    

    // Start is called before the first frame update
    void Start()
    {
        /*
        
        //string source = "/Users/emanuelazage/Documents/software_projects/Team-Scrubs/data/images/";
        //string destination = "/Users/emanuelazage/Documents/software_projects/Team-Scrubs/UnityProject/Assets/Resources/data/images/";
        string source = "/Users/emanuelazage/Documents/software_projects/Team-Scrubs/data/labels/";
        string destination = "/Users/emanuelazage/Documents/software_projects/Team-Scrubs/UnityProject/Assets/Resources/data/labels/";
        bool exists = Directory.Exists(source);
        Debug.Log("source folder exists: " + exists);
        if(exists){
            Directory.CreateDirectory(destination);   
            DirectoryInfo src = new FileInfo(source).Directory;
            DirectoryInfo dest = new FileInfo(destination).Directory;
            Debug.Log(src.FullName);
            foreach (var file in src.GetFiles()){
                file.CopyTo(Path.Combine(dest.FullName, file.Name), true);
                Debug.Log(file.Name);
            }
        }
        
         */

        allTextures = Resources.LoadAll<Texture2D>(resourcePath);
        
        Array.Sort(allTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });

        _renderer = GetComponent<Renderer>();

        _renderer.material.mainTexture = allTextures[imgNumber - 1];
        
        
        labelTextures = Resources.LoadAll<Texture2D>("data/label");
        
        Array.Sort(labelTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                var go = Instantiate(Brush, hit.point + Vector3.up * upMultiplier, Quaternion.identity, transform);
                go.transform.localScale = Vector3.one * BrushSize;
            }
        }

        if (_isRunning == 1)
        {
            StartCoroutine(updateUI());
        }
    }

    public IEnumerator updateUI()
    {
        _isRunning = 0;
        yield return new WaitForSeconds(_secFreqRefresh);
        updateUIItems();
        _isRunning = 0;
    }

    private void updateUIItems()
    {
        float similarityScore = evaluateAccuracy((Texture2D) _renderer.material.mainTexture, labelTextures[imgNumber-1]);
        
        Debug.Log(similarityScore);

    }

    public void updateBrushSize(float newSize)
    {
        BrushSize = newSize;
    }

    public void goToNextImage()
    {
        if (transform.childCount < 1)
        {
            Debug.Log("less than 1 child, can't go to next image until you shade in the cell");
            return;
        }
        
        StartCoroutine(saveAndNext());
        viewable.goToNextImage();
    }

    public void undo()
    {
        StartCoroutine(coUndo());
    }

    private IEnumerator coUndo()
    {
        yield return new WaitForEndOfFrame();

        // Deletes all brush strokes
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            GameObject.Destroy(transform.GetChild(i).gameObject);
        }
    }

    private IEnumerator saveAndNext()
    {
        yield return new WaitForEndOfFrame();

        // Save Image

        saveImg();

        if (imgNumber % 10 == 0)
        {
            newBatch();
        }

        imgNumber += 1;

        // Go To Next Image
        next();

    }

    private void saveImg()
    {
        
        RenderTexture.active = RTexture1;

        var texture2D = new Texture2D(RTexture1.width, RTexture1.height);
        texture2D.ReadPixels(new Rect(0, 0, RTexture1.width, RTexture1.height), 0, 0);

        //var imgData = texture2D.EncodeToPNG();

        //File.WriteAllBytes(Application.dataPath + "/Resources/data/drawings/" + imgNumber + ".png", imgData);

        var drawingPixels = texture2D.GetPixels();
        var maskOutputPixels = drawingPixels;

        for (int i = 0; i < drawingPixels.Length; i++)
        {
            if (drawingPixels[i].r >= 0.95 && drawingPixels[i].g < 0.6 && drawingPixels[i].b < 0.6) // is red
            {
                maskOutputPixels[i].a = 0;
            }
            else
            {
                maskOutputPixels[i].a = 1;
                maskOutputPixels[i].r = 0;
                maskOutputPixels[i].g = 0;
                maskOutputPixels[i].b = 0;
            }
        }

        var maskTexture = texture2D;
        maskTexture.SetPixels(maskOutputPixels);
        var pngData = texture2D.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/Resources/data/drawings/" + imgNumber + ".png", pngData);
        

        //Debug.Log(pxls[0]);Debug.Log(pxls[1]);Debug.Log(pxls[2]);Debug.Log(pxls[3]);
        //Debug.Log(pxls[256]);Debug.Log(pxls[257]);Debug.Log(pxls[258]);Debug.Log(pxls[259]);
        //Debug.Log(pxls[512]);Debug.Log(pxls[513]);Debug.Log(pxls[514]);Debug.Log(pxls[515]);
        //Debug.Log(pxls[768]);Debug.Log(pxls[769]);Debug.Log(pxls[770]);Debug.Log(pxls[771]);
        //Debug.Log(pxls[32382]);
    }

    public void newBatch()
    {
        //Go to cutscene
        Debug.Log("go to cut scene");
    }

    public void next()
    {
        // swap material texture
        
        _renderer.material.mainTexture = allTextures[imgNumber - 1];

        // Remove all brush stroke child objects
        foreach (Transform child in transform)
        {
            GameObject.Destroy(child.gameObject);
        }

    }
    
    private float evaluateAccuracy(Texture2D userInput, Texture2D groundTruth)
    {
        //Based on DICE metric on the assumption that texture pixels have alpha values of {0: background, 1: label} where alpha is "a" in Color(r,b,g,a)

        Color[] userPixels = userInput.GetPixels();
        Color[] gtPixels = groundTruth.GetPixels();

        float overlapArea = 0f;
        float userArea = 0f;
        float gtArea = 0f;

        for(int i=0; i<userPixels.Length; i++)
        {
            overlapArea += userPixels[i].a * gtPixels[i].a;
            userArea += userPixels[i].a;
            gtArea += gtPixels[i].a;
        }

        return (overlapArea * 2) / (userArea + gtArea);
    }

}

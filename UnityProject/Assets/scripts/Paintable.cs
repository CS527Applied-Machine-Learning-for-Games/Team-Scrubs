
using System;
using System.Collections;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public Slider sizeSlider;
    public RenderTexture RTexture1;
    public RenderTexture RTextureViewable;
    public TMP_Text scoreText;
    public Viewable viewable;
    
    public int imgNumber;
    public string paintableResourcePath;
    public string viewableResourcePath;
    
    private float _brushSize = 0.25f;
    private float _upMultiplier = 0f;

    private Texture2D[] allTextures;
    private Texture2D[] labelTextures;
    
    private Renderer _renderer;

    // automatic score update feature 
    private int _isRunning = 1;
    private int _secFreqRefresh = 2; // how often we update the ui - i.e. calculate score in seconds

    // undo button feature
    private int _undoDeleteRatio = 10;
    private int _undoSize;
    
    
    // Start is called before the first frame update
    void Start()
    {

        imgNumber = PlayerPrefs.GetInt("imgNum", 1);
        saveCurrentLocation();

        /*allTextures = Resources.LoadAll<Texture2D>(Application.streamingAssetsPath + paintableResourcePath);
        
        Array.Sort(allTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });*/
        string imageDirectory = Application.streamingAssetsPath + paintableResourcePath;

        string[] imageFiles = Directory.GetFiles(imageDirectory, "*png");

        allTextures = new Texture2D[imageFiles.Length];

        Array.Sort<string>(imageFiles,
            delegate (string x, string y) { return String.Compare(x, y, StringComparison.Ordinal); });

        //Does this need to work?

        /*Array.Sort(files, delegate (string x, string y) {
            return Int16.Parse(x).CompareTo(Int16.Parse(y));
        });*/

        for (int i = 0; i < imageFiles.Length; i++)
        {
            Texture2D tex = null;
            byte[] imageData;
            imageData = File.ReadAllBytes(imageFiles[i]);

            tex = new Texture2D(1, 1);
            tex.LoadImage(imageData);

            allTextures[i] = tex;
        }

        _renderer = GetComponent<Renderer>();

        _renderer.material.mainTexture = allTextures[imgNumber - 1];

        /*labelTextures = Resources.LoadAll<Texture2D>(Application.streamingAssetsPath + "/Resources/" + viewableResourcePath);
        
        Array.Sort(labelTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });*/

        string labelDirectory = Application.streamingAssetsPath + viewableResourcePath;

        string[] labelFiles = Directory.GetFiles(labelDirectory, "*png");

        labelTextures = new Texture2D[labelFiles.Length];

        Array.Sort<string>(labelFiles,
            delegate (string x, string y) { return String.Compare(x, y, StringComparison.Ordinal); });

        //Does this need to work?

        /*Array.Sort(labelFiles, delegate (string x, string y) {
            return Int16.Parse(x).CompareTo(Int16.Parse(y));
        });*/

        for (int i = 0; i < labelFiles.Length; i++)
        {
            Texture2D tex = null;
            byte[] imageData;
            imageData = File.ReadAllBytes(labelFiles[i]);

            tex = new Texture2D(1, 1);
            tex.LoadImage(imageData);

            labelTextures[i] = tex;
        }

        viewable.next(imgNumber); // updates label view as well
    }
    
    private void saveCurrentLocation()
    {
        string path = Application.streamingAssetsPath + "/data/player_data.txt";
        int currentImg = PlayerPrefs.GetInt("imgNum", 1);
        using (StreamWriter writer = new StreamWriter(path))  
        {  
            writer.WriteLine(currentImg.ToString());
        }
    }

    /*private void OnDestroy()
    {
        Debug.Log("clear player prefs on destroy");
        PlayerPrefs.DeleteAll();
    }*/

    // Update is called once per frame
    void Update()
    {

        var notOnUIControls = sizeSlider.fillRect.position.y - 60 > Input.mousePosition.y;
        if (Input.GetMouseButton(0) && notOnUIControls ) // putting down brush stokes
        {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                var go = Instantiate(Brush, hit.point + Vector3.up * _upMultiplier, Quaternion.identity, transform);
                go.transform.localScale = Vector3.one * _brushSize;
            }

            _undoSize = transform.childCount/_undoDeleteRatio;
        }

        if (_isRunning == 1) StartCoroutine(coUpdateUI());

        if (Input.GetKeyDown(KeyCode.UpArrow) && _brushSize < sizeSlider.maxValue) updateBrushSize(_brushSize + 0.1f); 
        if (Input.GetKeyDown(KeyCode.DownArrow) && _brushSize > sizeSlider.minValue) updateBrushSize(_brushSize - 0.1f);
        if (Input.GetKeyDown(KeyCode.RightArrow)){ goToNextImage(); }
        if (Input.GetKeyDown(KeyCode.LeftArrow)){ undo(); }

    }

    public IEnumerator coUpdateUI()
    {
        _isRunning = 0;
        yield return new WaitForSeconds(_secFreqRefresh);
        updateUI();
        _isRunning = 1;
    }

    private void updateUI()
    {
        RenderTexture.active = RTextureViewable;
        var viewableTexture = new Texture2D(RTextureViewable.width, RTextureViewable.height);
        viewableTexture.ReadPixels(new Rect(0, 0, RTextureViewable.width, RTextureViewable.height), 0, 0);

        float similarityScore = overlapScore(createMaskTexture(), viewableTexture);
        //Debug.Log("evaulateAcc: " + evaluateAccuracy(createMaskTexture(), processLabelForDiceEval(viewableTexture)).ToString());
        
        //Debug.Log("score: " + similarityScore.ToString());
        int roundedScore = (int)Math.Round(similarityScore, 0);
        scoreText.text = "Score " + roundedScore.ToString() + "%";

    }

    public void updateBrushSize(float newSize)
    {
        _brushSize = newSize;
        sizeSlider.value = _brushSize;
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

        // Deletes some ratio (_undoDeleteRatio) of highest number of brushstrokes before lifting paint brush
        for (int i = transform.childCount - 1; i >= Math.Max(transform.childCount-_undoSize, 0); i--)
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
            PlayerPrefs.SetInt("imgNum", imgNumber+1);
            newBatch();
        }
        Debug.Log("current imgNumber: " + imgNumber);
        imgNumber += 1;
        PlayerPrefs.SetInt("imgNum", imgNumber);
        // Go To Next Image
        next();

    }

    private void saveImg()
    {
        var maskTexture = createMaskTexture();
        var pngData = maskTexture.EncodeToPNG();
        
        File.WriteAllBytes(Application.streamingAssetsPath + "/data/drawings/" + imgNumber + ".png", pngData);
    }

    public void newBatch()
    {
        //Go to cutscene
        //DontDestroyOnLoad(imgNumber);
        SceneManager.LoadScene("Cutscene");
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
        
        saveCurrentLocation();
    }

    private float overlapScore(Texture2D drawnTexture, Texture2D labelTexture)
    {
        int maxOverlap = 0; // number of alpha = 0 pixels in gt
        int overLapCount = 0; // number of times same pixel in drawn img and in gt had alpha = 0

        Color[] drawnPixels = drawnTexture.GetPixels();
        Color[] labelPixels = labelTexture.GetPixels();

        if (drawnPixels.Length != labelPixels.Length)
        {
            Debug.Log("overlap score input lengths do not match!!");
            return -1;
        }

        for (int i = 0; i < labelPixels.Length; i++)
        {
            // TODO: Will want to convert label texture into a transparent mask and use this comparision instead of checking for dark pixels
            //if (labelPixels[i].a == 0)
            if (labelPixels[i].r > 0.05f)
            {
                maxOverlap += 1;
            }
        }

        for (int i = 0; i < drawnPixels.Length; i++)
        {
            // TODO: Will want to convert label texture into a transparent mask and use this comparision instead of checking for dark pixels
            //if (drawnPixels[i].a == 0 && labelPixels[i].a == 0) overLapCount += 1; // plus 1 for each pixel drawn that is in label mask
            //if (drawnPixels[i].a == 0 && labelPixels[i].a != 0) overLapCount -= 1; // minus 1 for each pixel drawn that is outside of label mask
            
            if (drawnPixels[i].r > 0.05f && labelPixels[i].r > 0.05f) overLapCount += 1; // plus 1 for each pixel drawn that is in label mask
            if (drawnPixels[i].r > 0.05f && labelPixels[i].r < 0.05f) overLapCount -= 1; // minus 1 for each pixel drawn that is outside of label mask
        }
        
        if (maxOverlap == 0) return 0;
        
        float score = ((float)overLapCount / (float)maxOverlap) * 100f;
        if (score < 0) score = 0;
        return score;

    }

    private Texture2D processLabelForDiceEval(Texture2D label)
    {
        Color[] labelPixels = label.GetPixels();
        
        for (int i = 0; i < labelPixels.Length; i++)
        {
            if (labelPixels[i].r > 0.05f)
            {
                labelPixels[i].a = 1;
            }
            else
            {
                labelPixels[i].a = 0;
            }
        }
        
        Texture2D output = new Texture2D(label.width, label.height);
        output.SetPixels(labelPixels);

        return output;
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

    private Texture2D createMaskTexture()
    {
        RenderTexture.active = RTexture1;

        var drawingTexture = new Texture2D(RTexture1.width, RTexture1.height);
        drawingTexture.ReadPixels(new Rect(0, 0, RTexture1.width, RTexture1.height), 0, 0);
        
        var drawingPixels = drawingTexture.GetPixels32();
        Texture2D maskTexture = new Texture2D(drawingTexture.width, drawingTexture.height);

        for (int i = 0; i < drawingTexture.width; i++)
        {
            for (int j = 0; j < drawingTexture.height; j++)
            {
                if (drawingPixels[i+ j * drawingTexture.width].r >= 0.95 && drawingPixels[i+ j * drawingTexture.width].g < 0.6 && drawingPixels[i+ j * drawingTexture.width].b < 0.6) // is red - the label pixels
                {
                    Color c = new Color(255, 255, 255, 1);
                    maskTexture.SetPixel(i, j, c);
                }
                else
                {
                    Color c = new Color(0, 0, 0, 1);
                    maskTexture.SetPixel(i, j, c);
                }
            }
        }

        
        
        return maskTexture;
    }

}


using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public Slider sizeSlider;
    public RenderTexture RTexture1;
    public RenderTexture RTextureViewable;
    public Viewable viewable;
    public int imgNumber = 1;
    public string paintableResourcePath;
    public string viewableResourcePath;
    
    private float _brushSize = 0.25f;
    private float _upMultiplier = 0f;

    private Texture2D[] allTextures;
    private Texture2D[] labelTextures;
    
    private Renderer _renderer;

    private int _isRunning = 1;
    private int _secFreqRefresh = 1;

    private int _undoDeleteRatio = 10;
    private int _undoSize;
    
    
    // Start is called before the first frame update
    void Start()
    {
        allTextures = Resources.LoadAll<Texture2D>(paintableResourcePath);
        
        Array.Sort(allTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });

        _renderer = GetComponent<Renderer>();

        _renderer.material.mainTexture = allTextures[imgNumber - 1];

        labelTextures = Resources.LoadAll<Texture2D>(viewableResourcePath);
        
        Array.Sort(labelTextures, delegate(Texture2D x, Texture2D y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });

    }

    // Update is called once per frame
    void Update()
    {
        
        if (Input.GetMouseButton(0) && Input.mousePosition.y < 730 ) // putting down brush stokes
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

        if (Input.GetKeyDown(KeyCode.UpArrow) && _brushSize < 1.0f) updateBrushSize(_brushSize + 0.1f); 
        if (Input.GetKeyDown(KeyCode.DownArrow) && _brushSize > 0.1f) updateBrushSize(_brushSize - 0.1f);
        
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

        float similarityScore = evaluateAccuracy(createMaskTexture(), viewableTexture);
        
        Debug.Log("score: " + similarityScore.ToString());

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
            newBatch();
        }

        imgNumber += 1;

        // Go To Next Image
        next();

    }

    private void saveImg()
    {
        var maskTexture = createMaskTexture();
        var pngData = maskTexture.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/Resources/data/drawings/" + imgNumber + ".png", pngData);
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

        for (int i = 0; i < drawnPixels.Length; i++)
        {
            if (drawnPixels[i].a == 0)
            {
                maxOverlap += 1;

                if (labelPixels[i].a == 0)
                {
                    overLapCount += 1;
                }
            }
        }

        return overLapCount / maxOverlap * 100; // dividing by zero
        
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
        
        var drawingPixels = drawingTexture.GetPixels();
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

        Texture2D maskTexture = new Texture2D(drawingTexture.width, drawingTexture.height);
        maskTexture.SetPixels(maskOutputPixels);

        return maskTexture;
    }

}

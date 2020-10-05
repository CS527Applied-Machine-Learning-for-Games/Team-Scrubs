using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Versioning;
using UnityEngine;
using UnityEngine.XR;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public RenderTexture RTexture1;

    public float BrushSize = 0.07f;
    public float upMultiplier = 0.05f;

    public int imgNumber = 1;
    public string resourcePath = "data/images";

    private Texture[] allTextures;

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

        allTextures = Resources.LoadAll<Texture>(resourcePath);
        
        Debug.Log("length1:" + allTextures.Length);
        Renderer renderer = GetComponent<Renderer>();

        renderer.material.mainTexture = allTextures[imgNumber - 1];
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
    }

    public void updateBrushSize(float newSize)
    {
        BrushSize = newSize;
        Debug.Log("brush size: " + BrushSize);
    }

    public void save()
    {
        StartCoroutine(saveAndNext());

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
            Debug.Log("check num: "+ imgNumber);
            newBatch();
        }

        imgNumber += 1;

        // Go To Next Image
        next();

    }

    private void saveImg()
    {
        //Debug.Log(Application.dataPath + "/createdImages/savedImage.png");

        RenderTexture.active = RTexture1;

        var texture2D = new Texture2D(RTexture1.width, RTexture1.height);
        texture2D.ReadPixels(new Rect(0, 0, RTexture1.width, RTexture1.height), 0, 0);

        var imgData = texture2D.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/createdImages/savedImage" + imgNumber + ".png", imgData);

    }

    public void newBatch()
    {
        //Go to cutscene
        Debug.Log("go to cut scene");

        //send the saved images to python and msg
        /*
        foreach (File in Folder)
        {
            file.CopyTo();
        }
        */
    }

    public void next()
    {
        // swap material texture
        Renderer renderer = GetComponent<Renderer>();
        //Debug.Log("length:"+allTextures.Length);
        renderer.material.mainTexture = allTextures[imgNumber - 1];

        // Remove all brush stroke child objects
        foreach (Transform child in transform)
        {
            GameObject.Destroy(child.gameObject);
        }
    }
}
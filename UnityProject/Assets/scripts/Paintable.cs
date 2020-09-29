using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public RenderTexture RTexture;
    
    public float BrushSize = 0.07f;
    public float upMultiplier = 0.05f;

    private GameObject[] brushStrokes = new GameObject[100000];
    
    // Start is called before the first frame update
    void Start()
    {
        
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

                brushStrokes.Append(go);
                
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
        StartCoroutine(coSave());
        
    }

    public void undo()
    {
        StartCoroutine(coUndo());
    }

    private IEnumerator coUndo()
    {
        yield return new WaitForEndOfFrame();
        Debug.Log("Deleting brush stokes");
        for (int i = 0; i < brushStrokes.Length; i++)
        {
            Destroy(brushStrokes[i]);
        }
    }
    
    private IEnumerator coSave()
    {
        yield return new WaitForEndOfFrame();
        Debug.Log(Application.dataPath + "/createdImages/savedImage.png");

        RenderTexture.active = RTexture;

        var texture2D = new Texture2D(RTexture.width, RTexture.height);
        texture2D.ReadPixels(new Rect(0, 0, RTexture.width, RTexture.height), 0, 0);

        var imgData = texture2D.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/createdImages/savedImage.png", imgData);
    }
}

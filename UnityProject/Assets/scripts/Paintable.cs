using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public RenderTexture RTexture1;
    
    public Texture[] textures;
    
    public Material material;
    
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
        for (int i = transform.childCount-1; i>=0; i--) {
            GameObject.Destroy(transform.GetChild(i).gameObject);
        }
    }
    
    private IEnumerator saveAndNext()
    {
        yield return new WaitForEndOfFrame();
        Debug.Log(Application.dataPath + "/createdImages/savedImage.png");

        RenderTexture.active = RTexture1;

        var texture2D = new Texture2D(RTexture1.width, RTexture1.height);
        texture2D.ReadPixels(new Rect(0, 0, RTexture1.width, RTexture1.height), 0, 0);

        var imgData = texture2D.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + "/createdImages/savedImage.png", imgData);

        // swap material texture
        Renderer renderer = GetComponent<Renderer>();

        var allTextures = Resources.LoadAll<Texture>("medical_images");
        //Texture loadedTexture = Resources.Load<Texture>("medical_images/Stevens_pancreatic_INS_1E_25mM_769_5_pre_rec_294");

        renderer.material.mainTexture = allTextures[0];
        
        // Remove all brush stroke child objects
        foreach (Transform child in transform) {
            GameObject.Destroy(child.gameObject);
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;

public class Paintable : MonoBehaviour
{
    public GameObject Brush;
    public RenderTexture RTexture;
    
    public float BrushSize = 0.05f;
    
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
                var go = Instantiate(Brush, hit.point + Vector3.up * 0.1f, Quaternion.identity, transform);
                go.transform.localScale = Vector3.one * BrushSize;
                var go2 = Instantiate(Brush, hit.point + Vector3.up * 0.1f, Quaternion.identity, transform);
                go2.transform.localScale = Vector3.one * BrushSize;
                var go3 = Instantiate(Brush, hit.point + Vector3.up * 0.1f, Quaternion.identity, transform);
                go3.transform.localScale = Vector3.one * BrushSize;
            }
        }
    }

    public void save()
    {
        StartCoroutine(coSave());
        
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

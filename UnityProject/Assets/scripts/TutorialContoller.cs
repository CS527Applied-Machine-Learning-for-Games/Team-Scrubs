using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TutorialContoller : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public float evaluateAccuracy(Texture2D userInput, Texture2D groundTruth)
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

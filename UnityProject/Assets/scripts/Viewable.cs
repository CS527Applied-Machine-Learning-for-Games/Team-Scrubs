﻿using System.Collections;
using UnityEngine;
using System;

public class Viewable : MonoBehaviour
{
    public int imgNumber = 1;
    public string resourcePath;

    private Texture[] allTextures;
    
    private Renderer _renderer;

    public Texture showHintTexture;
    

    // Start is called before the first frame update
    void Start()
    {

        allTextures = Resources.LoadAll<Texture>(resourcePath);

        Array.Sort<Texture>(allTextures,
            delegate(Texture x, Texture y) { return String.Compare(x.name,y.name, StringComparison.Ordinal); });
        
        Array.Sort(allTextures, delegate(Texture x, Texture y) {
            return Int16.Parse(x.name).CompareTo(Int16.Parse(y.name));
        });

        _renderer = GetComponent<Renderer>();

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


    }

    public void showHint()
    {
        _renderer.material.mainTexture = allTextures[imgNumber - 1];
    }
    

}


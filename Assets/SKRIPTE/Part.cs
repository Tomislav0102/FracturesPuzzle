using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using FirstCollection;

public class Part : MonoBehaviour
{
    internal GameManager gm;
    internal MeshRenderer meshRenderer;
    /*[HideInInspector]*/ public int ordinal;
    protected virtual void Ini()
    {
        gm = GameManager.gm;
        meshRenderer = GetComponent<MeshRenderer>();
        ordinal = transform.GetSiblingIndex();
    }
}

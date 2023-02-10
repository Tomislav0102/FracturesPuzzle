using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shadow : Part
{
    //Fragment _fragInPlace;
    //GameObject _connector;
    //public Fragment FragInPlace
    //{
    //    get => _fragInPlace;
    //    set
    //    {
    //        _fragInPlace = value;
    //        _connector.SetActive(value == null);
    //        gm.partsInPlace[ordinal] = value == null ? 99 : value.ordinal;
    //    }
    //}

    private void Awake()
    {
        Ini();
    }
    protected override void Ini()
    {
        base.Ini();
      //  _connector = transform.GetChild(0).gameObject;
    }
}

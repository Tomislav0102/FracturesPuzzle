using FirstCollection;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fragment : Part
{
    public FragState Fstate
    {
        get => _fstate;
        set
        {
            _fstate = value;
            _rigidBody.isKinematic = true;
            _myTransform.parent = gm.parFragment;
            switch (_fstate)
            {
                case FragState.Set:
                    _myTransform.parent = gm.parShadow;
                    break;
                case FragState.Idle:
                    _rigidBody.isKinematic = false;
                    break;
                case FragState.Carrying:
                    break;
                case FragState.InPlace:
                    _myTransform.parent = gm.parShadow;
                    break;
            }
        }
    }
    FragState _fstate;
    Transform _myTransform;
    Rigidbody _rigidBody;

    private void Awake()
    {
        Ini();
    }
    protected override void Ini()
    {
        base.Ini();
        _myTransform = transform;
        _rigidBody = GetComponent<Rigidbody>();

    }

    private void Update()
    {
        switch (Fstate)
        {
            case FragState.Set:
                break;
            case FragState.Idle:
                break;
            case FragState.Carrying:
                _myTransform.SetPositionAndRotation(gm.mousePos.position, gm.shadows[ordinal].transform.rotation);
                break;
        }
    }
    public void Explode(float force, Vector3 center)
    {
        _rigidBody.AddExplosionForce(force, center, 10f);
    }
    public void DropMe(int slotOrdinal)
    {
        switch (Fstate)
        {
            case FragState.Set:
                break;
            case FragState.Idle:
                break;
            case FragState.Carrying:
                if (slotOrdinal == 99)
                {
                    _rigidBody.AddExplosionForce(200f, _myTransform.position - Vector3.up, 2f);
                    Fstate = FragState.Idle;
                }
                else
                {
                    _myTransform.position = gm.shadows[slotOrdinal].transform.position;
                    if (slotOrdinal == ordinal)
                    {
                        Fstate = FragState.InPlace;
                    }
                    else Fstate = FragState.Set;
                }
                break;
        }


    }


}



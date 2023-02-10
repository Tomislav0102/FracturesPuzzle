using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using FirstCollection;

namespace SecondGame
{
    public class DragBehaviour : MonoBehaviour, IBeginDragHandler, IDragHandler
    {
        PictureManager pm;
        Vector2 _dir;
        MaskTile _maskTile;

        void Awake()
        {
            pm = PictureManager.pm;
            _maskTile = GetComponentInParent<MaskTile>();
        }

        //private void OnEnable()
        //{
        //    HelperScript.GameStart += CallEv_GameStart;
        //}
        //private void OnDisable()
        //{
        //    HelperScript.GameStart += CallEv_GameStart;
        //}
        //void CallEv_GameStart()
        //{

        //}

        public void OnBeginDrag(PointerEventData eventData)
        {
            _dir = eventData.delta;
            if (Mathf.Abs(_dir.x) > Mathf.Abs(_dir.y))
            {
                pm.MoveTile(_dir.x > 0f ? Direction.Right : Direction.Left, _maskTile.gridLocation);
            }
            else if (Mathf.Abs(_dir.x) < Mathf.Abs(_dir.y))
            {
                pm.MoveTile(_dir.y > 0f ? Direction.Up : Direction.Down, _maskTile.gridLocation);
            }
            else return;

        }

        public void OnDrag(PointerEventData eventData)
        {

        }
    }

}

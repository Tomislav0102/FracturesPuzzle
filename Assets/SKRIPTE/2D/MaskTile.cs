using FirstCollection;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace SecondGame
{
    public class MaskTile : MonoBehaviour
    {
        public Vector2Int ImageLocation
        {
            get => _imageLocation;
            set
            {
                _imageLocation = value;
                float pX = -_rectTransform.rect.width * (value.x + 0.5f);
                float pY = -_rectTransform.rect.height * (value.y + 0.5f);
                _img.GetComponent<RectTransform>().anchoredPosition = new Vector2(pX, pY);

            }
        }
        Vector2Int _imageLocation;
        public Vector2Int gridLocation;
        RectTransform _rectTransform;
        [SerializeField] Image _img;



        public void Ini(Vector2 kanvasSize, Vector2Int gridDim, Vector2Int gridPoz, Vector2Int imgPoz, Sprite sprite)
        {
            _rectTransform = GetComponent<RectTransform>();
            _rectTransform.anchoredPosition =
                new Vector2((kanvasSize.x / gridDim.x) * gridPoz.x - kanvasSize.x * 0.5f,
                (kanvasSize.y / gridDim.y) * gridPoz.y - kanvasSize.y * 0.5f);
            _rectTransform.sizeDelta = new Vector2(kanvasSize.x / gridDim.x, kanvasSize.y / gridDim.y);

            _img.sprite = sprite;
            _img.GetComponent<RectTransform>().sizeDelta = new Vector2(kanvasSize.x, kanvasSize.y);

            gridLocation = gridPoz;
            ImageLocation = imgPoz;
        }

    }

}

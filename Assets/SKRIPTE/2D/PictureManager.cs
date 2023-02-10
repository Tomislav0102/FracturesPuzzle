using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using FirstCollection;
using DG.Tweening;
using Coffee.UIEffects;
using UnityEngine.SceneManagement;
using TMPro;

namespace SecondGame
{
    public class PictureManager : MonoBehaviour
    {
        public static PictureManager pm;
        Vector2Int gridDim;
        [SerializeField] GameObject pictureFrame;
        [SerializeField] Canvas kanvas;
        [SerializeField] Transform gridTr;
        Vector2 kanSize;
        [SerializeField] MaskTile maskPrefab;
        MaskTile[,] _maskTiles;
        Transform[,] _maskTr;
        List<Vector2Int> _imgLoc = new List<Vector2Int>();
        List<Vector2Int> _rdnImgLoc = new List<Vector2Int>();
        [SerializeField] Sprite sprite;

        [SerializeField] RawImage finalImg;
        UITransitionEffect _finalImgEff;
        float _effTimer = 5f;
        bool _showImg;

        [SerializeField] GameObject overWindow;
        [SerializeField] Image overImage;
        [SerializeField] Button btnShowHide, btnRestart, btnQuit;

        bool _tweenCompleted = true;
        const float CONST_TWEENDURATION = 0.5f;
        [SerializeField] Ease izy;
        private void Awake()
        {
            pm = this;
            kanSize.x = kanvas.GetComponent<RectTransform>().rect.width;
            kanSize.y = kanvas.GetComponent<RectTransform>().rect.height;
            finalImg.texture = sprite.texture;
            finalImg.enabled = true;
            _finalImgEff = finalImg.GetComponent<UITransitionEffect>();
            overImage.sprite = sprite;
            btnShowHide.gameObject.SetActive(false);
            overWindow.SetActive(false);
        }
        private void Start()
        {
            gridDim = new Vector2Int(PlayerPrefs.GetInt("dimX"), PlayerPrefs.GetInt("dimY"));
            btnShowHide.transform.GetComponentInChildren<TextMeshProUGUI>().text = !_showImg ? "Prikazi sliku" : "Sakri sliku";

            StartCoroutine(IniGrid());
        }
        private void OnEnable()
        {
            btnShowHide.onClick.AddListener(Btn_ShowPicture);
            btnRestart.onClick.AddListener(Btn_Restart);
            btnQuit.onClick.AddListener(Btn_Quit);

            HelperScript.GameStart += CallEv_GameStart;
            HelperScript.GameOver += CallEv_GameOver;
        }
        private void OnDisable()
        {
            btnShowHide.onClick.RemoveAllListeners();
            btnRestart.onClick.RemoveAllListeners();
            btnQuit.onClick.RemoveAllListeners();

            HelperScript.GameStart -= CallEv_GameStart;
            HelperScript.GameOver -= CallEv_GameOver;
        }
        void CallEv_GameStart()
        {
            _finalImgEff.Hide();
            finalImg.raycastTarget = _showImg;
            btnShowHide.gameObject.SetActive(true);
        }
        void CallEv_GameOver()
        {
            _tweenCompleted = false;
            overWindow.SetActive(true);
            btnShowHide.gameObject.SetActive(false);
            pictureFrame.SetActive(false);
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Escape)) Btn_Quit();
        }
        IEnumerator IniGrid()
        {
            _maskTiles = new MaskTile[gridDim.x, gridDim.y];
            _maskTr = new Transform[gridDim.x, gridDim.y];
            for (int j = 0; j < gridDim.y; j++)
            {
                for (int i = 0; i < gridDim.x; i++)
                {
                    _imgLoc.Add(new Vector2Int(i, j));
                }
            }
            _rdnImgLoc = HelperScript.RandomListByType<Vector2Int>(_imgLoc);
           // _rdnImgLoc = _imgLoc;
            int counter = 0;
            for (int j = 0; j < gridDim.y; j++)
            {
                for (int i = 0; i < gridDim.x; i++)
                {
                    MaskTile mt = Instantiate(maskPrefab, gridTr);
                    _maskTiles[i, j] = mt;
                    _maskTr[i,j] = mt.transform;
                    mt.Ini(kanSize, gridDim, _imgLoc[counter], _rdnImgLoc[counter], sprite);
                    counter++;
                }
            }

            while(_effTimer >= 0f)
            {
                _effTimer -= Time.deltaTime;
                yield return null;
            }
            HelperScript.GameStart?.Invoke();
        }

        public void Btn_ShowPicture()
        {
            _finalImgEff.effectPlayer.duration = 0.5f;
            _showImg = !_showImg;
            btnShowHide.transform.GetComponentInChildren<TextMeshProUGUI>().text = !_showImg ? "Prikazi sliku" : "Sakri sliku";
            finalImg.raycastTarget = _showImg;
            if (_showImg) _finalImgEff.Show();
            else _finalImgEff.Hide();
        }
        public void Btn_Restart()
        {
            SceneManager.LoadScene(gameObject.scene.name);
        }
        public void Btn_Quit()
        {
            SceneManager.LoadScene(0);
        }

        public void MoveTile(Direction dir, Vector2Int startPos)
        {
            if (!_tweenCompleted) return;

            Vector2Int bufferImageLocation = _maskTiles[startPos.x, startPos.y].ImageLocation;
            Vector2Int bufferEndCoordinates = Vector2Int.zero;
            switch (dir)
            {
                case Direction.Up:
                    bufferEndCoordinates = startPos + Vector2Int.up;
                    break;

                case Direction.Right:
                    bufferEndCoordinates = startPos + Vector2Int.right;
                    break;

                case Direction.Down:
                    bufferEndCoordinates = startPos + Vector2Int.down;
                    break;

                case Direction.Left:
                    bufferEndCoordinates = startPos + Vector2Int.left;
                    break;
            }

            if (bufferEndCoordinates.x < 0 || bufferEndCoordinates.y < 0 || bufferEndCoordinates.x > gridDim.x - 1 || bufferEndCoordinates.y > gridDim.y - 1) return;

            _maskTiles[startPos.x, startPos.y].ImageLocation = _maskTiles[bufferEndCoordinates.x, bufferEndCoordinates.y].ImageLocation;
            _maskTiles[bufferEndCoordinates.x, bufferEndCoordinates.y].ImageLocation = bufferImageLocation;
            TweenAnimation(_maskTr[startPos.x, startPos.y], _maskTr[bufferEndCoordinates.x, bufferEndCoordinates.y]);
        }

        void TweenAnimation(Transform start, Transform end)
        {
            _tweenCompleted = false;

            Vector3 sp = start.position;
            Vector3 ep = end.position;
            start.DOMove(sp, CONST_TWEENDURATION)
                 .From(ep)
                 .SetEase(izy)
                 .OnComplete(CheckCompleted);
            end.DOMove(ep, CONST_TWEENDURATION)
               .From(sp)
               .SetEase(izy);
        }
        void CheckCompleted()
        {
            _tweenCompleted = true;

            for (int i = 0; i < gridDim.x; i++)
            {
                for (int j = 0; j < gridDim.y; j++)
                {
                    if (_maskTiles[i, j].gridLocation.x != _maskTiles[i, j].ImageLocation.x ||
                        _maskTiles[i, j].gridLocation.y != _maskTiles[i, j].ImageLocation.y) return;
                }
            }

            HelperScript.GameOver?.Invoke();
        }

    }

}

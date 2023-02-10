using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using FirstCollection;
using System;
using TMPro;
using UnityEngine.UI;

public class GameManager : MonoBehaviour
{
    public static GameManager gm;
    [SerializeField] GameObject kanMain, kanGameOver;
    [SerializeField] Button btnRestart, btnExit;
    [SerializeField] Material[] allMats;
    public Transform mousePos;

    public Transform parFragment, parShadow;
    Fragment[] _fragments;
    public Shadow[] shadows;
    int _fragCount;

    [SerializeField] LayerMask lmDef, lmSlots, lmFragments, lmMousePosDetect;
    [SerializeField] Camera mainCam;
    Transform _camParent, _camTransform;
    readonly Vector2 _camZoomBorders = new Vector2(0f, 6.4f);
    float _zoomValue;
    const float CONST_ZOOMSPEED = 60f;
    float _horRot;
    Vector3 mp;

    [SerializeField] float force, rotSpeed;
    [SerializeField] Transform center;

    Fragment _fragCarrying;
    int[] _partsInPlace;
    const float CONST_DISTTRESHHOLD = 1f;
    float[] _distances;
    Shadow _closestShadow;
    float _timerClosestShadow;

    bool _gameStarted;

    private void Awake()
    {
        gm = this;
    }
    private void Start()
    {
        _camTransform = mainCam.transform;
        _zoomValue = _camTransform.localPosition.z;
        _camParent = _camTransform.parent.parent;
        mp = Input.mousePosition;
        _fragments = HelperScript.GetAllChildernByType<Fragment>(parFragment);
        shadows = HelperScript.GetAllChildernByType<Shadow>(parShadow);
        kanMain.SetActive(false);
        kanGameOver.SetActive(false);
        _fragCount = _fragments.Length;
        _distances = new float[_fragCount];
        _partsInPlace = new int[_fragCount];
        for (int i = 0; i < _fragCount; i++)
        {
            _distances[i] = 99f;
            _partsInPlace[i] = 99;
        }
        ChangePart(99, MatChoice.Invisible);

        StartCoroutine(Expolsion());
    }
    private void OnEnable()
    {
        btnRestart.onClick.AddListener(Btn_Restart);
        btnExit.onClick.AddListener(Btn_Exit);

        HelperScript.GameStart += CallEv_GameStart;
        HelperScript.GameOver += CallEv_GameOver;
    }
    private void OnDisable()
    {
        btnRestart.onClick.RemoveAllListeners();
        btnExit.onClick.RemoveAllListeners();

        HelperScript.GameStart -= CallEv_GameStart;
        HelperScript.GameOver -= CallEv_GameOver;
    }
    void CallEv_GameStart()
    {
        kanMain.SetActive(true);
        _gameStarted = true;
    }
    void CallEv_GameOver()
    {
        kanMain.SetActive(false);
        kanGameOver.SetActive(true);
        _gameStarted = false;
        parShadow.rotation = Quaternion.identity;
    }
    void Btn_Restart()
    {
        SceneManager.LoadScene(gameObject.scene.name);
    }
    void Btn_Exit()
    {
        SceneManager.LoadScene(0);
    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape)) Btn_Exit();
        if (!_gameStarted) return;

        if (_fragCarrying != null) FindClosestShadow();

        mousePos.position = HelperScript.MousePoz(mainCam, lmMousePosDetect);




        if (Input.GetMouseButtonDown(0))
        {
            mp = Input.mousePosition;

            if (_fragCarrying != null)
            {
                if (_closestShadow != null)
                {
                    _fragCarrying.DropMe(_closestShadow.ordinal);
                    _partsInPlace[_closestShadow.ordinal] = _fragCarrying.ordinal;
                    ChangePart(_closestShadow.ordinal, MatChoice.Invisible);
                    CheckCompleted();
                }
                else _fragCarrying.DropMe(99);

                _fragCarrying = null;
                _closestShadow = null;
            }
            else
            {
                if (Physics.Raycast(mainCam.ScreenPointToRay(Input.mousePosition), out RaycastHit hit, 20f, lmFragments))
                {
                    if (hit.collider.TryGetComponent(out Fragment fr))
                    {
                        switch (fr.Fstate)
                        {
                            case FragState.Set:
                                for (int i = 0; i < _partsInPlace.Length; i++)
                                {
                                    if (fr.ordinal == _partsInPlace[i])
                                    {
                                        ChangePart(i, MatChoice.Transparent);
                                        _partsInPlace[i] = 99;
                                        break;
                                    }
                                }
                                break;
                            case FragState.Idle:
                                break;
                            case FragState.Carrying:
                                break;
                            case FragState.InPlace:
                                return;
                        }
                        _fragCarrying = fr;
                        _fragCarrying.Fstate = FragState.Carrying;
                       
                    }
                }

            }
        }

        if (Input.GetMouseButton(0) && _fragCarrying == null)
        {
            float diff = mp.x - Input.mousePosition.x;
            if (Mathf.Abs(diff) > 1f)
            {
                diff = Mathf.Clamp(diff, -1f, 1f);
                _horRot = diff * 0.5f;
            }
        }
        else _horRot = Input.GetAxis("Horizontal");


    }
    private void LateUpdate()
    {
        if (!_gameStarted) return;

        if (Input.mouseScrollDelta.y > 0) _zoomValue += Time.deltaTime * CONST_ZOOMSPEED;
        else if (Input.mouseScrollDelta.y < 0) _zoomValue -= Time.deltaTime * CONST_ZOOMSPEED;
        _zoomValue = Mathf.Clamp(_zoomValue, _camZoomBorders.x, _camZoomBorders.y);
        _camTransform.localPosition = _zoomValue * Vector3.forward;

        _camParent.Rotate(Time.deltaTime * rotSpeed * _horRot * Vector3.up);

    }
    private void FindClosestShadow()
    {
        _timerClosestShadow += Time.deltaTime;
        if (_timerClosestShadow > 0.2f) _timerClosestShadow = 0f;
        else return;

        float prevDist = Mathf.Infinity;
        int counter = 0;
        ChangePart(99, MatChoice.Transparent);
        for (int i = 0; i < _fragCount; i++)
        {
            if (_partsInPlace[i] == 99)
            {
                _distances[i] = Vector3.Distance(_fragCarrying.transform.position, shadows[i].transform.position);
                if (_distances[i] < prevDist)
                {
                    counter = i;
                    prevDist = _distances[i];
                }
            }
            else
            {
                _distances[i] = 99f;
                ChangePart(i, MatChoice.Invisible);
            }
        }

        if (_distances[counter] < CONST_DISTTRESHHOLD)
        {
            _closestShadow = shadows[counter];
            ChangePart(counter, MatChoice.Set);
        }
        else _closestShadow = null;
    }


    #region//MISC
    public void ChangePart(int ordinal, MatChoice mSet)
    {
        if (ordinal == 99)
        {
            for (int i = 0; i < shadows.Length; i++)
            {
                shadows[i].GetComponent<MeshRenderer>().material = allMats[(int)mSet];
            }
            return;
        }

        shadows[ordinal].GetComponent<MeshRenderer>().material = allMats[(int)mSet];
    }
    void CheckCompleted()
    {
        for (int i = 0; i < _partsInPlace.Length; i++)
        {
            if (_partsInPlace[i] != i) return; 
        }
        HelperScript.GameOver?.Invoke();
    }
    IEnumerator Expolsion()
    {
        yield return new WaitForSeconds(1f);
        for (int i = 0; i < _fragments.Length; i++)
        {
            _fragments[i].Fstate = FragState.Idle;
            _fragments[i].Explode(force, center.position);
        }
        ChangePart(99, MatChoice.Transparent);
        yield return new WaitForSeconds(2f);
        HelperScript.GameStart?.Invoke();
    }
    void ReAsemble()
    {
        for (int i = 0; i < _fragments.Length; i++)
        {
            _fragments[i].transform.SetPositionAndRotation(shadows[i].transform.position, shadows[i].transform.rotation);
            _fragments[i].Fstate = FragState.InPlace;
        }
        ChangePart(99, MatChoice.Invisible);
    }
    #endregion
}

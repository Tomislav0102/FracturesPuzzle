using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;

public class MainMenuManager : MonoBehaviour
{
    [SerializeField] GameObject mainMenu, slike;
    [SerializeField] Button btnSlike, btnSkulpture, btnExit, btnIgrajSlike, btnNatragGlIzbornik;
    [Header("Slike")]
    [SerializeField] TextMeshProUGUI[] displayDim;
    Button btnUpX, btnDownX, btnUpY, btnDownY;
    Vector2Int Dim
    {
        get => _dim;
        set
        {
            _dim = value;
            _dim.x = Mathf.Clamp(_dim.x, _dimLImits.x, _dimLImits.y);
            _dim.y = Mathf.Clamp(_dim.y, _dimLImits.x, _dimLImits.y);
            PlayerPrefs.SetInt("dimX", _dim.x);
            PlayerPrefs.SetInt("dimY", _dim.y);
            displayDim[0].text = _dim.x.ToString();
            displayDim[1].text = _dim.y.ToString();
        }
    }
    Vector2Int _dim;
    readonly Vector2Int _dimLImits = new Vector2Int(3, 10);

    private void Awake()
    {
        Btn_SlikeIzbornik(false);
        Dim = new Vector2Int(PlayerPrefs.GetInt("dimX"), PlayerPrefs.GetInt("dimY"));
        btnUpX = displayDim[0].transform.GetChild(1).GetComponent<Button>();
        btnDownX = displayDim[0].transform.GetChild(2).GetComponent<Button>();
        btnUpY = displayDim[1].transform.GetChild(1).GetComponent<Button>();
        btnDownY = displayDim[1].transform.GetChild(2).GetComponent<Button>();
    }

    private void OnEnable()
    {
        btnSlike.onClick.AddListener(delegate { Btn_SlikeIzbornik(true); });
        btnSkulpture.onClick.AddListener(delegate { Btn_Igraj(2); });
        btnExit.onClick.AddListener(Btn_Exit);
        btnIgrajSlike.onClick.AddListener(delegate { Btn_Igraj(1); });
        btnNatragGlIzbornik.onClick.AddListener(delegate { Btn_SlikeIzbornik(false); });

        btnUpX.onClick.AddListener(delegate { Btn_ChangeDim(true, true); });
        btnDownX.onClick.AddListener(delegate { Btn_ChangeDim(true, false); });
        btnUpY.onClick.AddListener(delegate { Btn_ChangeDim(false, true); });
        btnDownY.onClick.AddListener(delegate { Btn_ChangeDim(false, false); });
    }
    private void OnDisable()
    {
        btnSlike.onClick.RemoveAllListeners();
        btnSkulpture.onClick.RemoveAllListeners();
        btnExit.onClick.RemoveAllListeners();
        btnIgrajSlike.onClick.RemoveAllListeners();
        btnNatragGlIzbornik.onClick.RemoveAllListeners();

        btnUpX.onClick.RemoveAllListeners();
        btnDownX.onClick.RemoveAllListeners();
        btnUpY.onClick.RemoveAllListeners();
        btnDownY.onClick.RemoveAllListeners();
    }


    void Btn_SlikeIzbornik(bool otvoriSlike)
    {
        mainMenu.SetActive(!otvoriSlike);
        slike.SetActive(otvoriSlike);
    }
    void Btn_Igraj(int a)
    {
        SceneManager.LoadScene(a);
    }
    void Btn_Exit()
    {
        Application.Quit();
    }


    void Btn_ChangeDim(bool isHor, bool increment)
    {
        int num = increment ? 1 : -1;
        Dim = new Vector2Int(isHor ? Dim.x + num : Dim.x, !isHor ? Dim.y + num : Dim.y);
    }
}

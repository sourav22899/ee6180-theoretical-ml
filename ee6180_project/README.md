## An Empirical Study on Online Agnostic Boosting via Regret Minimization

This repo contains the code for the final project of EE6180 Advanced Topics in Artificial Intelliegence (Theoretical Machine Learning). In this project, we conduct an empirical study on the theoretical results of [Online Agnostic Boosting via Regret Minimization](https://papers.nips.cc/paper/2020/hash/07168af6cb0ef9f78dae15739dd73255-Abstract.html). Dataset 1 can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits) and Dataset 2 can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/isolet).

Create a directory `data` in the current directory and create two subdirectories in it namely `ocr` and `isolet`. Keep the datasets in the respective folders. To run the code for Dataset 1:
```
python main.py with ocr_config -p
```

For Dataset 2:
```
python main.py with isolet_config -p
```


## Requirements
* [`numpy`](http://www.numpy.org/)
* [`sklearn`](https://scikit-learn.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`tqdm`](https://tqdm.github.io/)
* [`sacred`](https://github.com/IDSIA/sacred)

These are all easily installable via, e.g., `pip install numpy` 


## Citation
If you find this repo useful in your research, please consider to cite this repo:


```
@misc{sahoo2020empirical,
  title={An Empirical Study on Online Agnostic Boosting via Regret Minimization},
  author={Sahoo, Sourav},
  howpublished={\url{https://github.com/sourav22899/ee6180-theoretical-ml/tree/master/ee6180_project}},
  year={2020}
}
```

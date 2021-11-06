# Model Risk in Credit Scoring

This repo is aimed at providing quantitative evidence how various stages of constructing a credit scoring model contribute to its performance.

_Financial Risk Management Lab, Higher School of Economics, Moscow, 2021-2022_


## Repo Structure

`/baseline/*` — Baseline (minimum viable product — MVP) model, proof of concept

`/datasets/*` — Overview of 16 most popular datasets employed in the credit scoring literature

`/scoring_model/*` — Modular architecture of all the credit scoring stages

`requirements.txt` lists all the libraries and versions to set up the virtual environment


## Modules Included

Overall pipeline is initially constructing using `sklearn.pipeline.Pipeline()` 
[docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

1. Preprocessing — many approaches already implemented in `feature-engine` library:
	+ missing values imputation
	+ categorical feature encoding
	+ variable transformation
	+ outlier detection and correction
2. Feature Engineering:
	+ featuretools [docs](https://featuretools.alteryx.com/en/stable/)
	+ Feature-engine — already supports sklearn-like `.fit()`/`.transform()` architecture [docs](https://feature-engine.readthedocs.io/en/1.1.x/)
3. Feature Selection
4. Additional Modules:
	+ Target imbalance correction
	+ Reject inference
5. Modelling:
	+ Hyperparameter optimization
6. Validation
	+ If trained on subsample: test if the sample is representative (PSI for sample vs. population)
	+ Gini index + confidence interval
	+ Discrimination stability (drops of quality — Train vs. Test OOS vs. Test OOT)
	+ Gini index for each model submodule
	+ Gini dynamics
	+ Factor contribution — Uplift test
	+ PD distribution — check if PDs are not concentrated in a small interval of values
	+ Calibration
	+ Precision-Recall curve stability
	+ Other metrics



* * *

# FAQ, Instructions & Useful Links

## 1. Как сделать клон GitHub- репозитория?

Вроде всем настроил, пишите лично. — _Антон_



## 2. Как поставить `venv`?

__Шаг 1.__ Открыть папку с клоном git-репозитория в терминале:

```console
cd C:\Users\antonmarkov\01-studies\01-phd\year-3\model-risks\credit_scoring_project
```

__Шаг 2.__ Установить `venv` в сооответствии с версиями библиотек в requirements.txt:

Windows-версия:

```console
python3 -m venv venv
./venv/Scripts/activate.bat
pip install -r requirements.txt
```

UNIX-версия:

```console
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```


## 3. Как запустить `venv`?

__Шаг 1.__ Открыть папку с клоном git-репозитория в терминале:

```console
cd C:\Users\antonmarkov\01-studies\01-phd\year-3\model-risks\credit_scoring_project
```

__Шаг 2 — Windows.__ Запустить `venv`:

```console
./venv/Scripts/activate.bat
```

__Шаг 2 — Linux / Mac.__ Запустить `venv`:

```console
source ./venv/bin/activate
```



## 4. Как сделать git-add-commit-push?

__Шаг 1.__ Открыть папку с клоном git-репозитория в терминале:

```console
cd C:\Users\antonmarkov\01-studies\01-phd\year-3\model-risks\credit_scoring_project
```

__Шаг 2.__ На всякий случай скчать все новые обновления с гитхаба (pull):

```console
git pull
```

__Шаг 3.__ Добавить новые файлы, папки и т.п. Данные на гитхаб не загружаем!

```console
git add new_file.ipynb
```

__Шаг 4.__ По всем добавленным файлам сохраняем текущую версию локально в version control (делаем commit). Не забываем указать осмысленное сообщение!

```console
git commit -m "I did this and that"
```

__Шаг 5.__ Отправляем коммит на гитхаб

```console
git push
```



## 5. Как работает Git?

1. Ссылка на ютуб-курс от Миши [здесь](https://www.youtube.com/watch?v=SEvR78OhGtw&t=4203s)

2. Хорошая книжка с документацией [здесь](https://git-scm.com/book/en/v2)

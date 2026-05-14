# 🎬 Movie Recommendation System — Matrix Factorization

> A personalized movie recommendation engine built **from scratch** using collaborative filtering and Stochastic Gradient Descent, trained on the MovieLens 1M dataset.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens%201M-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📖 Overview

This project implements a **model-based collaborative filtering** recommender system using **Vanilla Matrix Factorization (MF)** optimized with **Stochastic Gradient Descent (SGD)** — all built from scratch without high-level ML libraries like scikit-surprise or PyTorch.

Traditional content-based and memory-based collaborative filtering systems struggle to scale and personalize at the same time. This project addresses those limitations using the **latent factor approach** — the same core technique behind Netflix's prize-winning recommendation engine (Koren et al., 2009).

**BSc (Hons) Computing — Honours Dissertation**  
Edinburgh Napier University · August 2022

---

## ✨ Key Features

- **Pure NumPy implementation** — no high-level ML libraries; everything built from scratch
- **Personalized top-N recommendations** — unique ranked movie list per user based on rating history
- **Vanilla Matrix Factorization** — dot product of user matrix `P` and item matrix `Q`
- **SGD optimization** — iterative error minimization with L2 regularization
- **Early stopping** — training halts automatically when validation loss converges
- **Unseen-movie filtering** — only recommends movies the user has not yet rated
- **Three evaluation metrics** — MSE, RMSE, and MAE for full model assessment
- **Loss curve visualization** — training vs. validation loss plotted per epoch

---

## 📊 Results

| Metric | Score |
|--------|-------|
| MSE    | ~0.86 |
| RMSE   | ~0.92 |
| MAE    | ~0.73 |

**Sample output — Top 5 for User ID 6:**

| Rank | Movie Title                    | Predicted Rating |
|------|--------------------------------|-----------------|
| 1    | Crimes and Misdemeanors (1989) | 5.16            |
| 2    | Up in Smoke (1978)             | 5.11            |
| 3    | Stuart Little (1999)           | 5.08            |
| 4    | Natural, The (1984)            | 5.07            |
| 5    | Peter Pan (1953)               | 5.07            |

---

## 🗂️ Dataset

Uses the **[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)** from GroupLens Research.

| File | Description | Size |
|------|-------------|------|
| `ratings-cleaned.dat` | UserID, MovieID, Rating, Timestamp | ~1M ratings |
| `movies-cleaned.dat`  | MovieID, Title, Genre | 3,952 movies |
| `users-cleaned.dat`   | UserID, Gender, Age, Occupation, Zip | 6,040 users |

> Ratings are on a 1–5 star scale. Each user has at least 20 ratings.

---

## 🏗️ System Architecture

```
Database
   ├── ratings-cleaned.dat
   ├── movies-cleaned.dat
   └── users-cleaned.dat
         │
         ▼
   Data Splitting (60% train / 20% valid / 20% test)
         │
         ▼
   Matrix Factorization  ◄──── SGD Optimizer
         │                         │
         └──── Error Handling ─────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Evaluation (test)     Recommendation
   MSE / RMSE / MAE      (unseen movies, sorted)
```

---

## 💻 Core Implementation

### Matrix Factorization

```python
def Matrix_Factorization(P, Q, u, i):
    return np.dot(P[u, :], Q[i, :])
```

Rating prediction is the dot product of user latent vector `P[u]` and item latent vector `Q[i]`.

---

### SGD Optimizer

```python
def sgd(error, P, Q, id_user, id_item, features, lr, weigth_decay):
    for f in range(features):
        P[id_user, f] = P[id_user, f] + lr * (Q[id_item, f] * error - weigth_decay * P[id_user, f])
        Q[id_item, f] = Q[id_item, f] + lr * (P[id_user, f] * error - weigth_decay * Q[id_item, f])
    return P, Q
```

Updates both user and item matrices per observed rating. `weigth_decay` is L2 regularization to prevent overfitting.

---

### Training Loop with Early Stopping

```python
def errorhandling(data, features=10, lr=0.0002, epochs=101, weigth_decay=0.02, stopping=0.001):
    train = data[0]
    valid = data[1]
    numbers_of_users = len(train)
    numbers_of_items = len(train[0])
    loss_train, loss_valid = [], []

    P = np.random.rand(numbers_of_users, features) * 0.1
    Q = np.random.rand(numbers_of_items, features) * 0.1

    for e in range(epochs):
        for u in range(numbers_of_users):
            for i in range(numbers_of_items):
                if train[u][i] > 0:
                    error_ui = train[u][i] - Matrix_Factorization(P, Q, u, i)
                    P, Q = sgd(error_ui, P, Q, u, i, features, lr, weigth_decay)

        loss_train.append(MSE(train, P, Q))
        loss_valid.append(MSE(valid, P, Q))

        # Early stopping when validation loss stops improving
        if e > 1:
            if abs(loss_valid[-1] - loss_valid[-2]) < stopping:
                break

    return P, Q, loss_train, loss_valid
```

Best model found at around **epoch 47–49**. Train and validation loss curves converge cleanly with no overfitting.

---

### Evaluation Metrics

```python
def MSE(data, P, Q):
    sum_of_error, evaluation_count = 0., 0
    for u in range(len(data)):
        for i in range(len(data[0])):
            if data[u][i] > 0:
                sum_of_error += pow(data[u][i] - Matrix_Factorization(P, Q, u, i), 2)
                evaluation_count += 1
    return sum_of_error / evaluation_count

def RMSE(data, P, Q):
    sum_of_error, evaluation_count = 0., 0
    for u in range(len(data)):
        for i in range(len(data[0])):
            if data[u][i] > 0:
                sum_of_error += pow(data[u][i] - Matrix_Factorization(P, Q, u, i), 2)
                evaluation_count += 1
    return math.sqrt(sum_of_error / evaluation_count)

def MAE(data, P, Q):
    sum_of_error, evaluation_count = 0., 0
    for u in range(len(data)):
        for i in range(len(data[0])):
            if data[u][i] > 0:
                sum_of_error += abs(data[u][i] - Matrix_Factorization(P, Q, u, i))
                evaluation_count += 1
    return sum_of_error / evaluation_count
```

---

### Recommendation

```python
user_id = int(input("Enter User ID (1-6040):"))
topnumber = int(input("Enter Top Number movies:"))

user_train = np.array(train[0][user_id])
unseenmovies = np.where(user_train == 0, 1, 0)       # mask already-seen movies

MFPredict = np.dot(P[user_id, :], Q.T)               # predict ratings for all movies
unseen_movie_predict = MFPredict * unseenmovies        # zero out seen movies

recommendations, prediction_scores = top_rank(
    np.array(moviesdata['Titles']), unseen_movie_predict, k=topnumber
)
```

Only unseen movies are ranked and returned — the system never recommends a movie the user already rated.

---

## ⚙️ Hyperparameters

| Parameter      | Value  | Description                        |
|----------------|--------|------------------------------------|
| `features`     | 4      | Number of latent factors           |
| `lr`           | 0.001  | Learning rate                      |
| `epochs`       | 101    | Maximum training epochs            |
| `weigth_decay` | 0.01   | L2 regularization strength         |
| `stopping`     | 0.001  | Early stopping delta threshold     |

---

## 🚀 Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Data Setup

1. Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
2. Clean and place files under `data/ml-1M/`:
   - `ratings-cleaned.dat`
   - `movies-cleaned.dat`
   - `users-cleaned.dat`

### Run

```bash
jupyter notebook matrixfactorization.ipynb
```

Run cells in order. When prompted, enter a User ID (1–6040) and the number of top recommendations you want.

---

## 📁 Project Structure

```
movie-recommender-mf/
│
├── data/
│   └── ml-1M/
│       ├── ratings-cleaned.dat
│       ├── movies-cleaned.dat
│       └── users-cleaned.dat
│
├── Another.ipynb       # Main notebook (data loading, training, evaluation, recommendation)
├── utilities.py        # Helper functions
└── README.md
```

---

## ⚠️ Known Limitations

| Limitation     | Description |
|----------------|-------------|
| Cold start     | New users or new movies have no rating history to learn from |
| Data sparsity  | Users with very few ratings yield less accurate predictions |
| Speed          | Pure Python SGD loops are slow; not suitable for 10M+ datasets as-is |
| No UI          | Output is text/DataFrame only — no visual front-end |

---

## 🔭 Future Work

- **MF with side features** — add user demographics and implicit feedback to reduce cold start
- **DSGD** — replace standard SGD with distributed variant for larger datasets
- **Autoencoder** — upgrade to deep learning for better accuracy and noise resilience
- **Web interface** — build a proper UI so non-technical users can interact with the system

---

## 📚 References

- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer Society.
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TiiS.
- Le, J. & Ororbia, A. G. (2019). *From Matrix Factorization To Deep Neural Networks.* Rochester.

---

## 👤 Author

**Sai Bhone Myat Naing**  
BSc (Hons) Computing · Edinburgh Napier University · 2022

---

*Not Allowed to Reuse for any purpose*

# 📂 Data

This folder holds the MovieLens 1M dataset used for training and evaluation.

## Download Instructions

1. Go to: https://grouplens.org/datasets/movielens/1m/
2. Download `ml-1m.zip` and extract it
3. Place the cleaned files inside `data/ml-1M/` with these exact names:

```
data/
└── ml-1M/
    ├── ratings-cleaned.dat
    ├── movies-cleaned.dat
    └── users-cleaned.dat
```

## File Format

**ratings-cleaned.dat** — tab-separated
```
UserID  MovieID  Rating  Timestamp
1       1193     5       978300760
```

**movies-cleaned.dat** — tab-separated
```
MovieID  Title                  Genre
1        Toy Story (1995)       Animation|Children's|Comedy
```

**users-cleaned.dat** — tab-separated
```
UserID  Gender  Age  Occupation     Zip
1       F       1    other          48067
```

## Dataset Stats

| Property | Value |
|----------|-------|
| Ratings  | ~1,000,209 |
| Users    | 6,040 |
| Movies   | 3,952 |
| Scale    | 1–5 stars |

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015.
*The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
DOI: http://dx.doi.org/10.1145/2827872

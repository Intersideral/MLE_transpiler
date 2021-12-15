# Linear/Logistic Regression transpiler from joblib to C

## Installation

```
python3 -m venv venv
source ./venv/bin/activate
pip install requirements.txt
```

## Usage

### Generate a linear regression and a logistic regression model for test
```
python3 ./get_models.py
```

### Use the transpiler
```
python3 ./transpiler.py <filename> <is_logistic>
```
- \<filename\> path to your .joblib file
- \<is_logistic\> True if the model is a logistic regression, False if the model is a linear regression

### Compile the model
```
gcc model.c
```

### Use the model
```
./a.out <args>
```
Where \<args\> are the features that you want to predict

Exemple: ./a.out 2. 5.
# Linear Regression transpiler from joblib to C

## Installation

```
python3 -m venv venv
source ./venv/bin/activate
pip install requirements.txt
```

## Usage

### Generate a linear regression model for test
```
python3 ./get_model.py
```

### Use the transpiler
```
python3 ./transpiler.py <filename>
```
\<filename\> path to your .joblib file

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
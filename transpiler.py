import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser(description="Arguments for transpiler")
parser.add_argument("joblib", type=str, help="path of the model to transpile")
args = parser.parse_args()

model = joblib.load(args.joblib)
n_thetas = len(model.coef_) + 1
thetas = [model.intercept_]
thetas += model.coef_.tolist()
thetas = ','.join(np.array(thetas).astype(str))

with open("model.c", "w") as file:
    file.write("""
    #include <stdio.h>
    #include <stdlib.h>

    float linear_regression_prediction(float* features, float* thetas, int n_thetas) {
        float r = thetas[0];
        for (int i = 0; i < n_thetas - 1; i++)
        {
            r += features[i] * thetas[i+1];
        }
        return r;
    }
    """)
    file.write(f"""
    int main(int argc, char* argv[])
    {{
        float features[argc];
        for (int i = 1; i < argc; ++i)
        {{
            features[i - 1] = atof(argv[i]);
        }}
        int n_thetas = {n_thetas};
        float thetas[{n_thetas}] = {{ {thetas} }};
        float result = linear_regression_prediction(features, thetas, n_thetas);
        printf("%f\\n", result);
    }}
    """)
print("Compile with gcc model.c")
print("Expected result with 2, 5 :", model.predict([[2, 5]]))
import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser(description="Arguments for transpiler")
parser.add_argument("joblib", type=str, help="path of the model to transpile")
parser.add_argument("is_logistic", type=str, help="True if model is a logistic regression, False if model is a linear regression")
args = parser.parse_args()

model = joblib.load(args.joblib)
if args.is_logistic == "True":
    thetas = [model.intercept_[0]]
    thetas += model.coef_[0].tolist()
else:
    thetas = [model.intercept_]
    thetas += model.coef_.tolist()

n_thetas = len(thetas)
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

    float exp_approx(float x, int n_term) {
        float r = 1;
        float facto = 1;
        float power = x;
        for (int i = 1; i <= n_term; i++) {
            r += power / facto;
            power *= x;
            facto *= i + 1;
        }
        return r;
    }

    float sigmoid(float x) {
        float expo = exp_approx(-x, 10);
        return (float) 1 / (1 + expo);
    }
    """)
    if args.is_logistic == "True":
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
            result = sigmoid(result);
            printf("%f\\n", result);
        }}
        """)
    else:
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
if args.is_logistic == "True":
    print("Expected result with 1, 3 :", model.predict_proba([[1, 3]])[0,1])
else:
    print("Expected result with 2, 5 :", model.predict([[2, 5]]))

    


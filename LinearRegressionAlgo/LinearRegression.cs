using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinearRegressionAlgo
    
{
  
        class LinearRegression
        {
            double learning_rate;
            int iterations;
            int m, n;
            double[] W;
            double b;
            double[,] X, Y;

            public LinearRegression(double learning_rate, int iterations)
            {
                this.learning_rate = learning_rate;
                this.iterations = iterations;
            }

            public LinearRegression fit(double[,] X, double[,] Y)
            {
                this.X = X;
                this.Y = Y;
                this.m = X.GetLength(0);
                this.n = X.GetLength(1);
                this.W = new double[n];
                this.b = 0;

                for (int i = 0; i < this.iterations; i++)
                {
                    this.update_weights();
                }

                return this;
            }

            public double[,] predict(double[,] X)
            {
                int m = X.GetLength(0);
                int n = X.GetLength(1);
                double[,] Y_pred = new double[m, 1];

                for (int i = 0; i < m; i++)
                {
                    double sum = 0;

                    for (int j = 0; j < n; j++)
                    {
                        sum += X[i, j] * this.W[j];
                    }

                    Y_pred[i, 0] = sum + this.b;
                }

                return Y_pred;
            }

            void update_weights()
            {
                double[,] Y_pred = this.predict(this.X);

                double[,] dW = new double[this.n, 1];
                double db = 0;

                for (int i = 0; i < this.n; i++)
                {
                    double sum = 0;

                    for (int j = 0; j < this.m; j++)
                    {
                        sum += this.X[j, i] * (this.Y[j, 0] - Y_pred[j, 0]);
                    }

                    dW[i, 0] = -2 * sum / this.m;
                    this.W[i] -= this.learning_rate * dW[i, 0];
                }

                double sum2 = 0;

                for (int i = 0; i < this.m; i++)
                {
                    sum2 += (this.Y[i, 0] - Y_pred[i, 0]);
                }

                db = -2 * sum2 / this.m;
                this.b -= this.learning_rate * db;
            }
        }
    
}

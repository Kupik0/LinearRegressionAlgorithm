using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class LinearRegression
    {
        public double learning_rate;
        public int iterations;
        public int m, n;
        public double[] W;
        public double b;
        public double[,] X, Y;

        public LinearRegression(double learning_rate, int iterations)
        {
            this.learning_rate = learning_rate;
            this.iterations = iterations;
        }

        public LinearRegression fit(double[,] X, double[,] Y) // modelin eğitimi
        {
            this.X = X; //deneyim
            this.Y = Y; //maaş
            this.m = X.GetLength(0);// deneyim adet
            this.n = X.GetLength(1); // deneyim başlık
            this.W = new double[n]; // kaç başlık olduğuna göre katsayı(a)
            this.b = 0; // (b)

            for (int i = 0; i < this.iterations; i++) //optimize dönüş sayısı
            {
                this.update_weights();//a yı güncelleyelim
            }

            return this;
        }

        public double[,] predict(double[,] X) // tahminleri oluşturalım (Y için)
        {
            int m = X.GetLength(0); //adet
            int n = X.GetLength(1); // başlık
            double[,] Y_pred = new double[m, 1]; // deneyim verisinin adeti kadar y tahmini oluşturalım

            for (int i = 0; i < m; i++) // her veri için bir işlem
            {
                double sum = 0;

                for (int j = 0; j < n; j++) // eğer başlıkları 1 den fazlaysa katsayıyı(a) ona göre ekleyelim
                {
                    sum += X[i, j] * this.W[j];//tahmini değerimizi oluşturuyoruz
                }

                Y_pred[i, 0] = sum + this.b;// b yi ekliyoruz ve i'nci deneyim için tahmin ettiğimiz maaş
            }

            return Y_pred;
        }

        void update_weights()
        {
            double[,] Y_pred = this.predict(this.X); // Deneyimlerimizi(X[n]) gönderelim bize tahmini Maaş(Y[n]) değerleri dönsünler

            double[,] dW = new double[this.n, 1];// x verilerinin katsayıyla çarpımı ve gerçek verilerin katsayıyla çarpımı arasındaki fark için
            double db = 0; // b nin gerçek değerlerle arasındaki fark

            for (int i = 0; i < this.n; i++) // x başlıkları için yani 1 kereliğine
            {
                double sum = 0;

                for (int j = 0; j < this.m; j++) // her deneyim için birer kere
                {
                    sum += this.X[j, i] * (this.Y[j, 0] - Y_pred[j, 0]); // gerçek y değerlerinden tahmini y değerleri çıkarılıp X değerleri ile çarpılıyor
                }

                dW[i, 0] = -2 * sum / this.m; // sonucu 1/2 m ile çarpınca dw yi buluyoruz  ( 1/2 türevini alırken karenin gitmesi için)
                this.W[i] -= this.learning_rate * dW[i, 0]; //learning rate ile bulduğumuz değerin katsayımızı etkileme boyutunu ayarlıyoruz
            }

            double sum2 = 0;

            for (int i = 0; i < this.m; i++) // aynı şeyler b değeri için
            {
                sum2 += (this.Y[i, 0] - Y_pred[i, 0]);
            }

            db = -2 * sum2 / this.m;
            this.b -= this.learning_rate * db;
        }
    }
}

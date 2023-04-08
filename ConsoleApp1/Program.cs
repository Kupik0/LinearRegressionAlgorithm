
using ConsoleApp1;

Console.WriteLine("Deneyim ve maaşa göre veri tahmini");
double[,] X = new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } };
double[,] Y = new double[,] { { 3000 }, { 3500 }, { 4000 }, { 4500 }, { 5000 } };


LinearRegression model = new LinearRegression(0.01, 1000);
model.fit(X, Y);

double[,] X_test = { { 7 }, { 30 }, { 8 } };
double[,] Y_pred = model.predict(X_test);

Console.WriteLine("Katsayılar: ");
for (int i = 0; i < model.W.Length; i++)
{
    Console.WriteLine("W" + (i + 1) + ": " + model.W[i]);
}
Console.WriteLine("Eklemeler: " + model.b);

Console.WriteLine("\nTahminler: ");
for (int i = 0; i < X_test.GetLength(0); i++)
{
    Console.WriteLine("X" + (i + 1) + ": " + X_test[i, 0] + " => Y_pred: " + Y_pred[i, 0]);
}
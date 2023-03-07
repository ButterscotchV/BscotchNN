using System;
using BscotchNN;
using BscotchNN.Activation;
using BscotchNN.Error;
using MathNet.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace BscotchNNTests
{
    [TestClass]
    public class NetworkTests
    {
        private static readonly double DIST_SEPARATOR = 0.5d;
        private static readonly double DIST_SEPARATOR_SQR = DIST_SEPARATOR * DIST_SEPARATOR;

        private double GetLengthSqr(double x, double y)
        {
            return x * x + y * y;
        }

        private double GetLength(double x, double y)
        {
            return Math.Sqrt(GetLengthSqr(x, y));
        }

        private double GetOtherComponent(double r, double x)
        {
            return Math.Sqrt(r * r - x * x);
        }

        [TestMethod]
        public void NetworkTest()
        {
            Control.UseBestProviders();

            var network = new Network(2);

            // Hidden layers
            network.MakeLayer(4, TanhActivation.Singleton);
            network.MakeLayer(16, TanhActivation.Singleton);
            network.MakeLayer(128, TanhActivation.Singleton);
            network.MakeLayer(128, TanhActivation.Singleton);
            network.MakeLayer(128, TanhActivation.Singleton);
            network.MakeLayer(128, TanhActivation.Singleton);
            network.MakeLayer(16, TanhActivation.Singleton);
            network.MakeLayer(2, TanhActivation.Singleton);

            // Output
            network.MakeLayer(1, TanhActivation.Singleton);

            network.ApplyKaiserInit();

            var numIters = 100000;
            var printEvery = numIters / 10;
            var lossSum = 0.0d;
            var lossCount = 0;

            var random = new Random();
            for (var i = 0; i < numIters; i++)
            {
                var radius = random.NextDouble();

                var valX = random.NextDouble() * radius;
                var valY = GetOtherComponent(radius, valX);

                var predictions = network.Propagate(new[] { valX, valY });
                var loss = network.Backpropagate(new[] { radius * radius }, SquareError.Singleton);
                lossSum += loss;
                lossCount++;

                if (i % printEvery == 0)
                {
                    Console.WriteLine(
                        $"Iter: {i}, In: [{valX}, {valY}], Out: {predictions[0]}, Loss: {(lossCount > 0 ? lossSum / lossCount : -1.0)}");
                    lossSum = 0.0d;
                    lossCount = 0;
                }

                network.ApplyErr();
            }
        }
    }
}

using System;
using BscotchNN;
using BscotchNN.Activation;
using BscotchNN.Error;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
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
            network.MakeLayer(16, TanhActivation.Singleton);
            network.MakeLayer(2, TanhActivation.Singleton);

            // Output
            network.MakeLayer(1, TanhActivation.Singleton);

            network.ApplyKaiserInit();

            var numIters = 10000;
            var numEpochs = 100;

            var printEvery = numEpochs / 20;
            
            var learnRate = 0.5d;
            var decayRate = 1.0d;

            var random = new Random();
            for (var epoch = 0; epoch < numEpochs; epoch++)
            {
                var epochLearnRate = (1 / (1 + decayRate * epoch)) * learnRate;

                var lossSum = 0.0d;
                var lossCount = 0;

                for (var iter = 0; iter < numIters; iter++)
                {
                    var radius = random.NextDouble();

                    var valX = random.NextDouble() * radius;
                    var valY = GetOtherComponent(radius, valX);

                    var predictions = network.Propagate(new[] { valX, valY });
                    var loss = network.Backpropagate(Vector<double>.Build.Dense(new[] { radius * radius }), SquareError.Singleton);
                    lossSum += loss;
                    lossCount++;

                    network.ApplyErr(epochLearnRate);
                }

                var curEpoch = epoch + 1;
                if (curEpoch <= 1 || curEpoch % printEvery == 0)
                {
                    Console.WriteLine($"Epoch: {curEpoch}/{numEpochs} ({(curEpoch * 100.0d) / numEpochs:0.00}%), Loss: {(lossCount > 0 ? lossSum / lossCount : -1.0):0.0#######}, LR: {epochLearnRate:0.0#######}");
                }
            }
        }
    }
}

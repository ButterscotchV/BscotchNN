using System;
using MathNet.Numerics.LinearAlgebra;

namespace BscotchNN.Layers
{
    public class Layer
    {
        public readonly Vector<double> neurons;

        public ConnectedLayer? child;

        public Layer(int numNeurons)
        {
            if (numNeurons <= 0)
                throw new ArgumentOutOfRangeException(nameof(numNeurons), numNeurons,
                    $"{nameof(numNeurons)} must be greater than 0");

            neurons = Vector<double>.Build.Dense(numNeurons);
        }
    }
}

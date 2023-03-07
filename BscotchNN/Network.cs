using System;
using System.Collections.Generic;
using System.IO;
using BscotchNN.Activation;
using BscotchNN.Error;
using BscotchNN.Layers;

namespace BscotchNN
{
    public class Network
    {
        public readonly Layer inputLayer;
        public readonly List<ConnectedLayer> connectedLayers = new List<ConnectedLayer>();

        public Layer EndLayer => connectedLayers.Count > 0 ? connectedLayers[^1] : inputLayer;
        public ConnectedLayer OutputLayer => connectedLayers[^1];

        public Network(int numInputNeurons)
        {
            inputLayer = new Layer(numInputNeurons);
        }

        private static readonly Random InitRandom = new Random();
        private static double GenerateNormalRandom()
        {
            return Math.Sqrt(-2.0 * Math.Log(1.0 - InitRandom.NextDouble())) *
                   Math.Sin(2.0 * Math.PI * (1.0 - InitRandom.NextDouble()));
        }

        public void ApplyNormalInit()
        {
            foreach (ConnectedLayer layer in connectedLayers)
                unsafe
                {
                    fixed (double* pConnectionWeights = layer.connectionWeights.AsArray())
                    {
                        for (var i = 0; i < layer.connectionWeights.AsArray().Length; i++)
                            pConnectionWeights[i] = GenerateNormalRandom();
                    }
                }
        }

        private const double SqrtTwo = 1.41421356237;
        public void ApplyKaiserInit()
        {
            foreach (ConnectedLayer layer in connectedLayers)
            {
                var neuronCount = layer.connectionWeights.RowCount;
                var kaiserVal = SqrtTwo / Math.Sqrt(neuronCount);

                for (var x = 0; x < neuronCount; x++)
                    for (var y = 0; y < layer.connectionWeights.ColumnCount; y++)
                        layer.connectionWeights[x, y] = GenerateNormalRandom() * kaiserVal;
            }
        }

        public ConnectedLayer MakeLayer(int numNeurons, IActivation activation, double dropout = 0.0)
        {
            var layer = new ConnectedLayer(numNeurons, EndLayer, activation, dropout);
            connectedLayers.Add(layer);

            return layer;
        }

        public Layer MakeLayer(int numNeurons, double dropout = 0.0)
        {
            return MakeLayer(numNeurons, LinearActivation.Singleton, dropout);
        }

        public double[] Propagate(double[] inputs)
        {
            if (inputs.Length != inputLayer.neurons.Count)
                throw new ArgumentException(
                    $"The length of {nameof(inputs)} must be the same as the number of input neurons", nameof(inputs));

            // Set the first layer's values
            for (var i = 0; i < inputLayer.neurons.Count; i++) inputLayer.neurons[i] = inputs[i];

            foreach (var layer in connectedLayers)
                layer.Forward();

            return OutputLayer.neurons.AsArray();
        }

        public void Backpropagate(double[] errorDerivatives)
        {
            var outputLayer = OutputLayer;

            if (errorDerivatives.Length != outputLayer.neurons.Count)
                throw new ArgumentException(
                    $"The length of {nameof(errorDerivatives)} must be the same as the number of output neurons",
                    nameof(errorDerivatives));

            outputLayer.AddError(errorDerivatives);

            for (var i = connectedLayers.Count - 2; i >= 0; i--) connectedLayers[i].Backward();
        }

        public double Backpropagate(double[] expectedOutputs, IError error)
        {
            Layer outputLayer = OutputLayer;

            if (expectedOutputs.Length != outputLayer.neurons.Count)
                throw new ArgumentException(
                    $"The length of {nameof(expectedOutputs)} must be the same as the number of output neurons",
                    nameof(expectedOutputs));

            // Calculate the last layer's error values and calculate loss
            double loss = 0;
            var errorDerivatives = new double[expectedOutputs.Length];
            for (var i = 0; i < outputLayer.neurons.Count; i++)
            {
                loss += error.Calculate(outputLayer.neurons[i], expectedOutputs[i]);
                errorDerivatives[i] = error.Derivative(outputLayer.neurons[i], expectedOutputs[i]);
            }

            Backpropagate(errorDerivatives);

            return loss;
        }

        public double CalculateLoss(double[] expectedOutputs, IError error)
        {
            Layer outputLayer = OutputLayer;

            if (expectedOutputs.Length != outputLayer.neurons.Count)
                throw new ArgumentException(
                    $"The length of {nameof(expectedOutputs)} must be the same as the number of output neurons",
                    nameof(expectedOutputs));

            // Calculate loss for each output
            double loss = 0;
            for (var i = 0; i < outputLayer.neurons.Count; i++)
                loss += error.Calculate(outputLayer.neurons[i], expectedOutputs[i]);

            return loss;
        }

        public void ApplyErr(double learningRate = 1)
        {
            foreach (var layer in connectedLayers)
                layer.ApplyError(learningRate);
        }

        public void Save(string path)
        {
            BinaryWriter writer = new BinaryWriter(File.Create(path));

            try
            {
                foreach (var layer in connectedLayers)
                {
                    unsafe
                    {
                        fixed (double* pConnectionWeights = layer.connectionWeights.AsArray())
                        {
                            for (var i = 0; i < layer.connectionWeights.AsArray().Length; i++)
                                writer.Write(pConnectionWeights[i]);
                        }
                    }

                    foreach (var neuronBias in layer.neuronBiases) writer.Write(neuronBias);
                }
            }
            finally
            {
                writer.Close();
            }
        }

        public void Load(string path)
        {
            BinaryReader reader = new BinaryReader(File.OpenRead(path));

            try
            {
                foreach (var layer in connectedLayers)
                {
                    unsafe
                    {
                        fixed (double* pConnectionWeights = layer.connectionWeights.AsArray())
                        {
                            for (var i = 0; i < layer.connectionWeights.AsArray().Length; i++)
                                pConnectionWeights[i] = reader.ReadDouble();
                        }
                    }

                    for (var i = 0; i < layer.neuronBiases.Length; i++) layer.neuronBiases[i] = reader.ReadDouble();
                }
            }
            finally
            {
                reader.Close();
            }
        }
    }
}

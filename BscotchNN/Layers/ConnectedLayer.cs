using System;
using BscotchNN.Activation;
using MathNet.Numerics.LinearAlgebra;

namespace BscotchNN.Layers
{
    public class ConnectedLayer : Layer
    {
        public readonly Layer parent;
        public IActivation activation;

        public static readonly Random DropoutRandom = new Random();
        public double dropout;

        protected readonly Vector<double> rawNeurons;

        public readonly double[] neuronBiases;
        public readonly double[] neuronErrors;

        public readonly Matrix<double> connectionWeights;
        public readonly Matrix<double> connectionErrors;

        public readonly Matrix<double> outerProductCache;

        public int numErrors = 0;

        public ConnectedLayer(int numNeurons, Layer parent, IActivation activation, double dropout = 0.0) :
            base(numNeurons)
        {
            this.parent = parent;
            parent.child = this;

            this.activation = activation;

            this.dropout = dropout;

            rawNeurons = Vector<double>.Build.Dense(numNeurons);

            neuronBiases = new double[numNeurons];
            neuronErrors = new double[numNeurons];

            connectionWeights = Matrix<double>.Build.Dense(numNeurons, parent.neurons.Count);
            connectionErrors = Matrix<double>.Build.Dense(numNeurons, parent.neurons.Count);

            outerProductCache = Matrix<double>.Build.Dense(numNeurons, parent.neurons.Count);
        }

        private bool ShouldDropNeuron()
        {
            if (dropout <= 0.0)
                return false;

            if (dropout >= 1.0)
                return true;

            return DropoutRandom.NextDouble() < dropout;
        }

        public void Forward()
        {
            connectionWeights.Multiply(parent.neurons, neurons);

            for (var i = 0; i < neurons.Count; i++)
            {
                var inputValue = neurons[i] + neuronBiases[i];
                rawNeurons[i] = inputValue;

                neurons[i] = ShouldDropNeuron() ? 0.0 : activation.Calculate(inputValue);
            }
        }

        public void AddError(Vector<double> errors)
        {
            // Pass the error to the neurons
            for (var i = 0; i < rawNeurons.Count; i++)
            {
                var neuronError = errors[i] * activation.Derivative(rawNeurons[i]);

                rawNeurons[i] = neuronError;
                neuronErrors[i] += neuronError;
            }

            rawNeurons.OuterProduct(parent.neurons, outerProductCache);
            connectionErrors.Add(outerProductCache, connectionErrors);

            numErrors++;
        }

        public void Backward()
        {
            if (child == null)
                throw new NullReferenceException($"{nameof(child)} must be defined to propagate upstream");

            AddError(child.connectionWeights.TransposeThisAndMultiply(child.rawNeurons));
        }

        public void ApplyError(double learningRate)
        {
            // Apply the error to the neuron biases
            for (var i = 0; i < neuronBiases.Length; i++)
            {
                neuronBiases[i] -= (learningRate * neuronErrors[i]) / numErrors;
                neuronErrors[i] = 0;
            }

            connectionErrors.Multiply(learningRate / numErrors, connectionErrors);
            connectionWeights.Subtract(connectionErrors, connectionWeights);
            connectionErrors.Clear();

            numErrors = 0;
        }
    }
}

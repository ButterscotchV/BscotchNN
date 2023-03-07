using System;

namespace BscotchNN.Activation
{
    public class SigmoidActivation : IActivation
    {
        public static readonly SigmoidActivation Singleton = new SigmoidActivation();

        public double Calculate(double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }

        public double Derivative(double val)
        {
            var output = Calculate(val);
            return output * (1 - output);
        }
    }
}

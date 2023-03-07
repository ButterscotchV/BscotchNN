using System;

namespace BscotchNN.Activation
{
    public class TanhActivation : IActivation
    {
        public static readonly TanhActivation Singleton = new TanhActivation();

        public double Calculate(double val)
        {
            if (double.IsPositiveInfinity(val)) return 1;

            if (double.IsNegativeInfinity(val)) return -1;

            var e2x = Math.Exp(2 * val);
            return (e2x - 1) / (e2x + 1);
        }

        public double Derivative(double val)
        {
            var output = Calculate(val);
            return 1 - output * output;
        }
    }
}

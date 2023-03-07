namespace BscotchNN.Activation
{
    public class LeakyReluActivation : IActivation
    {
        public static readonly LeakyReluActivation Singleton = new LeakyReluActivation(0.01F);

        public readonly double leakRate;

        public LeakyReluActivation(double leakRate)
        {
            this.leakRate = leakRate;
        }

        public double Calculate(double val)
        {
            return val < 0 ? val * leakRate : val;
        }

        public double Derivative(double val)
        {
            return val < 0 ? leakRate : 1;
        }
    }
}

namespace BscotchNN.Activation
{
    public class LeakyReLUActivation : IActivation
    {
        public static readonly LeakyReLUActivation Singleton = new LeakyReLUActivation(0.01F);

        public readonly double leakRate;

        public LeakyReLUActivation(double leakRate)
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

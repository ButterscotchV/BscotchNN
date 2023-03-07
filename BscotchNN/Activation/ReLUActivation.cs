namespace BscotchNN.Activation
{
    public class ReLUActivation : IActivation
    {
        public static readonly ReLUActivation Singleton = new ReLUActivation();

        public double Calculate(double val)
        {
            return val < 0 ? 0 : val;
        }

        public double Derivative(double val)
        {
            return val < 0 ? 0 : 1;
        }
    }
}

namespace BscotchNN.Activation
{
    public class LinearActivation : IActivation
    {
        public static readonly LinearActivation Singleton = new LinearActivation();

        public double Calculate(double val)
        {
            return val;
        }

        public double Derivative(double val)
        {
            return 1;
        }
    }
}

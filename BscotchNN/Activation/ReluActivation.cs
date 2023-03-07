namespace BscotchNN.Activation
{
    public class ReluActivation : IActivation
    {
        public static readonly ReluActivation Singleton = new ReluActivation();

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

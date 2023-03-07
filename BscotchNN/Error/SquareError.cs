namespace BscotchNN.Error
{
    public class SquareError : IError
    {
        public static readonly SquareError Singleton = new SquareError();

        public double Calculate(double val, double target)
        {
            var error = val - target;
            return 0.5F * error * error;
        }

        public double Derivative(double val, double target)
        {
            return val - target;
        }
    }
}

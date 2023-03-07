namespace BscotchNN.Error
{
    public interface IError
    {
        double Calculate(double val, double target);
        double Derivative(double val, double target);
    }
}

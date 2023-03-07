namespace BscotchNN.Activation
{
    public interface IActivation
    {
        double Calculate(double val);
        double Derivative(double val);
    }
}

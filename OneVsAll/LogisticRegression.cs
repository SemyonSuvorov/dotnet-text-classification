using Deedle;
using Deedle.Math;
using MathNet.Numerics.LinearAlgebra;
using TextPreprocessing;

namespace ClassLibrary;

public class LogisticRegression
{
    private Vector<double>? _weights;
    private readonly int _maxIterations;
    private int _numOfFeatures;
    
    public LogisticRegression(int maxIterations)
    {
        _maxIterations = maxIterations;
    }
    
    public void Fit(Matrix<double> featureMatrix, Series<int,double> yTrue, int predictedClass, double learningRate)
    {
        //save target from dataframe
        //predict class 1 as positive, others as negative
        var y = yTrue
            .Select(x => (int)x.Value == predictedClass ? 1.0 : 0.0).ToVector();
        
        _numOfFeatures = featureMatrix.ColumnCount;
        //initialize weights
        var temp = new double[featureMatrix.ColumnCount];
        var nr = new NormalRandom();
        for (int i = 0; i < temp.Length; i++)
        {
            temp[i] = nr.NextDouble();
        }
        _weights = Vector<double>.Build.DenseOfArray(temp);
        
        //train cycle
        var losses = new List<double>();
        for (int i = 0; i < _maxIterations; i++)
        {
            if(i % 10 == 0) Console.WriteLine($"Regressor for {predictedClass} class, training progress: {i}%");
            //count dot-product
            var yPred = (featureMatrix * _weights).Map(Sigmoid,Zeros.Include);
            var loss = ComputeLoss(y, yPred);
            losses.Add(loss);
            GradientDescent(featureMatrix,  y, yPred, learningRate);
        }
        Console.WriteLine($"LogLoss for {predictedClass} class: {losses.First()} -> {losses.Last()}");
    }

    public double PredictProbaForOneSample(string input, PreprocessText p)
    {
        //vectorize input
        var feature = p.VectorizeOneFeature(input);
        //check shapes
        if (feature.Count != _numOfFeatures)
        {
            throw new Exception($"Shape of feature vector must be equal to {_numOfFeatures}");
        }
        //add bias term
        var yPred = Sigmoid(feature * _weights);
        return yPred;
    }

    private void GradientDescent(Matrix<double> featureMatrix,
        Vector<double> yTrue,Vector<double> yPred, double lr)
    {
        //logloss derivative
        var grad = (yPred - yTrue) * featureMatrix / featureMatrix.RowCount;
        //update weights
        _weights -= lr * grad;
    }

    private static double ComputeLoss(IEnumerable<double> yTrue, IList<double> yPred)
    {
        //count loss for each value
        var a = yTrue.Select((t, i) => LogLoss(t, yPred[i])).ToList();
        return a.Sum();
    }
    private static double LogLoss(double yTrue, double pred)
    {
        var yPred = Math.Clamp(pred, 1e-10, 1 - 1e-10);
        return -(yTrue * Math.Log(yPred) + (1 - yTrue) * Math.Log(1 - yPred));
    }
    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
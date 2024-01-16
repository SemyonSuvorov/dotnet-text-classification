using Deedle;
using Deedle.Math;

using MathNet.Numerics.LinearAlgebra;


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
    
    public void Fit(Frame<int, string> trainDf, double learningRate)
    {
        _numOfFeatures = trainDf.ColumnCount - 1;
        //initialize weights
        var temp = new double[trainDf.ColumnCount - 1];
        var nr = new NormalRandom();
        for (int i = 0; i < temp.Length; i++)
        {
            temp[i] = nr.NextDouble();
        }
        _weights = Vector<double>.Build.DenseOfArray(temp);
        
        //save target from dataframe
        var y = trainDf.Columns["Target"].Select(x => Convert.ToDouble(x.Value)).ToVector();

        var featureMatrix = ConvertFrameToFeatureMatrix(trainDf);
        
        //train cycle
        var losses = new List<double>();
        for (int i = 0; i < _maxIterations; i++)
        {
            //count dot-product
            var yPred = (featureMatrix * _weights).Map(Sigmoid,Zeros.Include);
            var loss = ComputeLoss(y, yPred);
            losses.Add(loss);
            GradientDescent(featureMatrix,  y, yPred, learningRate);
        }
        
        Console.WriteLine($"LogLoss: {losses.First()} -> {losses.Last()}");
    }

    public List<double> PredictProbaForMultipleSamples(Frame<int, string> testFrame)
    {

        var featureMatrix = ConvertFrameToFeatureMatrix(testFrame);
        var yPred = (featureMatrix * _weights).Map(Sigmoid,Zeros.Include);
        return yPred.ToList();
    }
    
    public double PredictProbaForOneSample(List<double> feature)
    {
        //check shapes
        if (feature.Count != _numOfFeatures)
        {
            throw new Exception($"Shape of feature vector must be equal to {_numOfFeatures}");
        }
        //add bias term
        feature[0] = 1.0;
        var featureVector = Vector<double>.Build.DenseOfEnumerable(feature);
        var yPred = Sigmoid(featureVector * _weights);
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
    private static double LogLoss(double yTrue, double yPred)
    {
        return -(yTrue * Math.Log(yPred) + (1 - yTrue) * Math.Log(1 - yPred));
    }
    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private static Matrix<double> ConvertFrameToFeatureMatrix(Frame<int, string> frame)
    {
        //drop useless columns
        frame.DropColumn("Target");
        frame.DropColumn("Column1");
        //convert to matrix
        var features = Matrix.ofFrame(frame);
        
        //create and add bias term
        var ones = new double[frame.RowCount];
        Array.Fill(ones, 1.0);
        var bias = Vector<double>.Build.DenseOfArray(ones);
        var featureMatrix = features.InsertColumn(0, bias);
        
        return featureMatrix;

    }
}
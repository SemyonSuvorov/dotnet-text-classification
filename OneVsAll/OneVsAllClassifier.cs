using ClassLibrary;
using Deedle;
using MathNet.Numerics.LinearAlgebra;
using TextPreprocessing;

namespace OneVsAll;

public class OneVsAllClassifier
{
    private static Series<int, double> _yTrue;
    private readonly List<LogisticRegression> _regressors = new();
    private readonly Matrix<double> _featureMatrix;
    private bool _isTrained;
    
    public OneVsAllClassifier(Frame<int, string> trainDf, int maxIterations)
    {
        //save target
        _yTrue = trainDf.Columns["Target"]
            .Select(x => Convert.ToDouble(x.Value));
        //add regressions for each class
        for (var i = 0; i < 4; i++)
        {
            _regressors.Add(new LogisticRegression(maxIterations));
        }
        //drop target and vectorize features
        trainDf.DropColumn("Target");
        _featureMatrix = PreprocessText.VectorizeFeaturesOneVsAll(trainDf);
    }

    public void Train(double learningRate)
    {
        Console.WriteLine("Started training...");
        Parallel.For(0, _regressors.Count, i =>
        {
            _regressors[i].Fit(_featureMatrix, _yTrue, (i + 1), learningRate);
        });
        _isTrained = true;
        Console.WriteLine("Done!");
        Console.WriteLine();

    }

    public int PredictClassForSample(string input)
    {
        if (!_isTrained) throw new Exception("Model is not trained");
        var probas = new double[4];
        var predictedClass = 0;
        var maxProba = double.MinValue;
        for (var i = 0; i < 4; i++)
        {
            probas[i] = _regressors[i].PredictProbaForOneSample(input);
            if (!(probas[i] > maxProba)) continue;
            maxProba = probas[i];
            predictedClass = i + 1;
        }
        return predictedClass;
    }
}
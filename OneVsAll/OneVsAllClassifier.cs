using ClassLibrary;
using Deedle;
using MathNet.Numerics.LinearAlgebra;
using TextPreprocessing;

namespace OneVsAll;

public class OneVsAllClassifier
{
    public static Series<int, double> _yTrue;
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
        Task task1 = Task.Run(() => _regressors[0].Fit(_featureMatrix, _yTrue, 1, learningRate));
        Task task2 = Task.Run(() => _regressors[1].Fit(_featureMatrix, _yTrue, 2, learningRate));
        Task task3 = Task.Run(() => _regressors[2].Fit(_featureMatrix, _yTrue, 3, learningRate));
        Task task4 = Task.Run(() => _regressors[3].Fit(_featureMatrix, _yTrue, 4, learningRate));
        Task.WaitAll(task1, task2, task3, task4);
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
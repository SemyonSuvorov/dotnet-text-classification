using Deedle;
using MathNet.Numerics.LinearAlgebra;
using TextPreprocessing;

namespace ClassLibrary;

public class OneVsAllClassifier
{
    private readonly Series<int, double> _yTrue;
    private List<LogisticRegression> _regressors = new();
    private readonly PreprocessText _p = new();
    private Matrix<double> _featureMatrix;
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
        _featureMatrix = _p.VectorizeFeatures(trainDf);
    }

    public void Train(double learningRate)
    {
        int j = 0;
        for (int i = 1; i < 5; i++)
        {
            _regressors[j].Fit(_featureMatrix, _yTrue, i, learningRate);
            j++;
        }
        _isTrained = true;
    }

    public int PredictClassForSample(string input)
    {
        if (!_isTrained) throw new Exception("Model is not trained");
        var probas = new double[4];
        int predictedClass = 0;
        double maxProba = double.MinValue;
        for (int i = 0; i < 4; i++)
        {
            probas[i] = _regressors[i].PredictProbaForOneSample(input, _p);
            if (probas[i] > maxProba)
            {
                maxProba = probas[i];
                predictedClass = i + 1;
            }
        }
        return predictedClass;
    }
}
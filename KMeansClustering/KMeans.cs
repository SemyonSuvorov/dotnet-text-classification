using Deedle;
using MathNet.Numerics.LinearAlgebra;
using Deedle.Math;
using OneVsAll;
using TextPreprocessing;
using System.Data;

namespace KMeans;


public class KMeans
{
    private readonly double[] _yTrue;
    private readonly double[][] _dataSet;
    private readonly double[] _assigments;
    private readonly List<Centroid> _centroidList;
    private readonly int _k;

    public KMeans(Frame<int, string> trainDf, int k)
    {
        _k = k;
        _yTrue = OneVsAllClassifier._yTrue.ToVector().ToArray();
        _dataSet = PreprocessText.GetFeaturesForKMeans(trainDf).ToRowArrays();
        _centroidList = new List<Centroid>();
        _assigments = new double[_yTrue.Length];
        CreateCentroids();
    }
    private void CreateCentroids()
    {
        for (int i = 0; i < _k; i++)
        {
            var centroid = new Centroid(_dataSet);
            _centroidList.Add(centroid);
        }

        Console.WriteLine("Training KMeans clustering...");

        while (true)
        {
            foreach (Centroid centroid in _centroidList) centroid.Reset();

            for (int i = 0; i < _dataSet.GetLength(0); i++)
            {
                double[] point = _dataSet[i];
                int closestIndex = -1;
                double minDistance = double.MaxValue;
                for (int k = 0; k < _centroidList.Count; k++)
                {
                    double distance = EuclideanDistance(_centroidList[k].Array, point);
                    if (distance < minDistance)
                    {
                        closestIndex = k;
                        minDistance = distance;
                    }
                }
                _centroidList[closestIndex].AddPoint(point);
            }
            foreach (Centroid centroid in _centroidList)
            {
                centroid.MoveCentroid();
            }
            Console.WriteLine("Centroids are moved.");

            bool hasChanged = false;
            foreach (Centroid centroid in _centroidList)
            {
                if (centroid.HasChanged())
                {
                    hasChanged = true;
                    break;
                }
            }

            if (!hasChanged)
            {
                break;
            }

            
        }
        AssignPoints();
        Console.WriteLine("Done!");
        Console.WriteLine();
    }

    private void AssignPoints()
    {
        int index = 0;
        foreach (var point in _dataSet)
        {
            _assigments[index] = Classify(point);
            index++;
        }
    }

    public int Classify(double[] input)
    {
        int closestIndex = -1;
        double minDistance = double.MaxValue;
        for (int k = 0; k < _centroidList.Count; k++)
        {
            double distance = EuclideanDistance(_centroidList[k].Array, input);
            if (distance < minDistance)
            {
                closestIndex = k;
                minDistance = distance;
            }
        }
        return closestIndex;
    }

    public double ClassifyString(string s)
    {
        var input = PreprocessText.VectorizeOneFeatureKMeans(s);
        int closestIndex = -1;
        double minDistance = double.MaxValue;
        for (int k = 0; k < _centroidList.Count; k++)
        {
            double distance = EuclideanDistance(_centroidList[k].Array, input);
            if (distance < minDistance)
            {
                closestIndex = k;
                minDistance = distance;
            }
        }
        return RecognizeCluster(closestIndex);
    }

    private double RecognizeCluster(int cluster)
    {
        var s = Vector<double>.Build.DenseOfArray(_assigments);
        var temp = Matrix<double>.Build
            .DenseOfColumns(new IEnumerable<double>[] { _assigments, _yTrue }).ToRowArrays();
        var filteredMatrix = temp.Where(row => (int)row[0] == (int)cluster);
        var tempMat = Matrix<double>.Build.DenseOfRowArrays(filteredMatrix);
        return tempMat.Column(1).ToList().GroupBy(x => x).MaxBy(x => x.Count())!.Key;
    }
    
    public static double EuclideanDistance(double[] firstPoint, double[] secondPoint)
    {
        var res = firstPoint.Select((t, i) => Math.Pow(t - secondPoint[i], 2)).Sum();
        return Math.Sqrt(res);
    }

}
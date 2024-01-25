using Deedle;
using MathNet.Numerics.LinearAlgebra;
using Deedle.Math;
using OneVsAll;
using TextPreprocessing;

namespace KMeans;


public class KMeans
{
    private readonly Vector<double> _yTrue;
    private readonly Matrix<double> _featureMatrix;
    private Vector<double> _assigments;
    private readonly Matrix<double> _centroids;
    private readonly List<double> _clusters;

    public KMeans(Frame<int, string> trainDf, int k)
    {
        _yTrue = OneVsAllClassifier._yTrue.ToVector();
        _featureMatrix = PreprocessText.GetFeaturesForKMeans();
        Console.WriteLine("Training KMeans clustering...");
        var centroids = Centroid.CreateCentroids(_featureMatrix, k);
        var assigments = AssignPoints(centroids);

        Vector<double>? oldAssigments = null;
        while (!assigments.Equals(oldAssigments))
        {
            var newCentroids = Centroid.UpdateCentroids(_featureMatrix, assigments, k);
            oldAssigments = assigments;
            assigments = AssignPoints(newCentroids);
            Console.WriteLine("Centroids are moved.");
            _centroids = newCentroids;
        }
        Console.WriteLine("Done!");
        Console.WriteLine();
        _assigments = assigments;
        _clusters = assigments.Distinct().Select(x => x + 1).ToList();
    }

    private double RecognizeCluster(double cluster)
    {
        var temp = Matrix<double>.Build
            .DenseOfColumns(new IEnumerable<double>[] { _assigments, _yTrue }).ToRowArrays();
        var filteredMatrix = temp.Where(row => (int)row[0] == (int)cluster);
        var tempMat = Matrix<double>.Build.DenseOfRowArrays(filteredMatrix);
        return tempMat.Column(1).ToList().GroupBy(x => x).MaxBy(x => x.Count())!.Key;
    }
    
    public double PredictCluster(string input)
    {
        var feature = PreprocessText.VectorizeOneFeature(input, "kmeans");
        var resultIndex = -1;
        var minDistance = double.MaxValue;
        for (var clusterIndex = 0; clusterIndex < _clusters.Count; clusterIndex++)
        {
            var dist = MathNet.Numerics.Distance.Euclidean(feature, _centroids.Row(clusterIndex));
            if (!(dist < minDistance)) continue;
            minDistance = dist;
            resultIndex = clusterIndex;
        }
        return resultIndex;
    }
    private Vector<double> AssignPoints(Matrix<double> centroids)
    {
        var assigments = Vector<double>.Build.Dense(_featureMatrix.RowCount);
        for (var pointIndex = 0; pointIndex < _featureMatrix.RowCount; pointIndex++)
        {
            var closestCentroidIndex = 0;
            var minDistance = double.MaxValue;
            for (var centroidIndex = 0; centroidIndex < centroids.RowCount; centroidIndex++)
            {
                var distance = MathNet.Numerics.Distance.Euclidean(_featureMatrix.Row(pointIndex), centroids.Row(centroidIndex));
                if (!(distance < minDistance)) continue;
                minDistance = distance;
                closestCentroidIndex = centroidIndex;
            }

            assigments[pointIndex] = closestCentroidIndex;
        }

        return assigments;
    }
    
    private static double EuclideanDistance(Vector<double> firstPoint, Vector<double> secondPoint)
    {
        var res = firstPoint.Select((t, i) => Math.Pow(t - secondPoint[i], 2)).Sum();
        return Math.Sqrt(res);
    }

}
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace KMeans;

public static class Centroid
{
    private static readonly Random Random = new();
    private static List<int> _centroidsIndexes = new();
    
    public static Matrix<double> CreateCentroids(Matrix<double> dataset, int k)
    {
        var result = Matrix<double>.Build.Dense(k, dataset.ColumnCount);
        for (int i = 0; i < k; i++)
        {
            result.SetRow(i, RandomPoint(dataset));
        }

        return result;
    }
    private static Vector<double> MeanPoint(Matrix<double> points)
    {
        var result = Vector<double>.Build.Dense(points.ColumnCount);
        for (int i = 0; i < points.ColumnCount; i++)
        {
            result[i] = points.Column(i).Mean();
        }

        return result;
    }

    public static Matrix<double> UpdateCentroids(Matrix<double> dataset, Vector<double> assigments, int k)
    {
        var clusterInds = assigments.Distinct().ToList();
        var newCentroids = Matrix<double>.Build
            .Dense(assigments.Distinct().Count(), dataset.ColumnCount);
        var assigmentsDataset = dataset.InsertColumn(dataset.ColumnCount, assigments).ToRowArrays();
        for (var clusterIndex = 0; clusterIndex < clusterInds.Count; clusterIndex++)
        {
            var clusterPoints = assigmentsDataset
                .Where(row => (int)row[^1] == clusterInds[clusterIndex]);
            var m = Matrix<double>.Build
                .DenseOfRowArrays(clusterPoints).RemoveColumn(dataset.ColumnCount);
            var newCentroid = MeanPoint(m);
            newCentroids.SetRow(clusterIndex, newCentroid);
        }
        return newCentroids;
    }
    
    private static Vector<double> RandomPoint(Matrix<double> points)
    {
        var index = Random.Next(0, points.RowCount);
        while (_centroidsIndexes.Contains(index))
        {
            index = Random.Next(0, points.RowCount);
        }
        _centroidsIndexes.Add(index);
        var randomPoint = points.Row(index);
        return randomPoint;
    }
}
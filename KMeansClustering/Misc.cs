

namespace KMeans
{
    public static class Misc
    {
        static Misc()
        {

        }

        public static List<double[]> Clone(List<double[]> array)
        {
            var resultList = new List<double[]>();
            foreach (double[] tempArray in array)
            {
                double[] newArray = new double[tempArray.Length];
                for (int i = 0; i < tempArray.Length; i++)
                    newArray[i] = tempArray[i];
                resultList.Add(newArray);
            }
            return resultList;
        }

        public static List<Tuple<double, double>> GetMinMaxPoints(double[][] dataset)
        {
            var result = new List<Tuple<double, double>>();

            for (int j = 0; j < dataset[0].GetLength(0); j++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;
                for (int i = 0; i < dataset.Length; i++)
                {
                    double element = dataset[i][j];
                    if (element < min)
                        min = element;
                    if (element > max)
                        max = element;
                }
                result.Add(new Tuple<double, double>(min, max));
            }
            return result;
        }
    }
}


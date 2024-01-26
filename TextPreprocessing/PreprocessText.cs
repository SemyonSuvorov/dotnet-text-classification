using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using Aspose.Cells;
using Deedle;
using static Microsoft.FSharp.Core.ByRefKinds;
using System.Globalization;


namespace TextPreprocessing;

public static partial class PreprocessText
{
    private static int _numOfDocs;
    private static Dictionary<string, double> _idfDict = new();
    private static readonly Dictionary<string, int> DfDict = new();
    private static readonly List<string> StopWordsList = new();
    private static Matrix<double>? _featureMatrix;
    private static Matrix<double>? _kMeansFeatureMatrix;
    private static Dictionary<string, int> _kMeansWordFreqDict = new();

    static PreprocessText()
    {
        const string filePath = "stopwords.txt";
        var stream = new FileStream(filePath, FileMode.Open);
        var reader = new StreamReader(stream);
        var parts = reader.ReadLine()?.Split(',');

        if (parts != null)
            foreach (var t in parts)
            {
                StopWordsList.Add(t);
            }

        reader.Close();
    }
    private static Dictionary<string, int> TokenizeAndTf(string input)
    {
        var tokens = input.Split(" ").ToList();
        var tfDict = new Dictionary<string, int>();
        List<string> listOfTokens = new();
        
        foreach (var token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = MyRegex().Replace(loweredToken, "")
                .Replace(@"\", "").Replace("\"","");
            
            if (StopWordsList.Contains(normalizedToken)) continue;
            
            if (tfDict.TryGetValue(normalizedToken, out int value)) tfDict[normalizedToken] = ++value;
            
            else tfDict.TryAdd(normalizedToken, 1);
            
            if (!listOfTokens.Contains(normalizedToken)) listOfTokens.Add(normalizedToken);
        }
        foreach (var tok in listOfTokens)
        {
            if (DfDict.TryGetValue(tok, out var value)) DfDict[tok] = ++value;
            
            else DfDict.TryAdd(tok, 1);
        }
        return tfDict;
    }

    private static Dictionary<string, double> ComputeIdf()
    {
        return DfDict.ToDictionary(pair => pair.Key, pair => Math.Log(_numOfDocs / pair.Value));
    }

    public static Vector<double> VectorizeOneFeature(string input)
    {
        var tokens = input.Split(" ").ToList();
        var tfDict = new Dictionary<string, int>();
        //tokenize 
        foreach (var token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = MyRegex().Replace(loweredToken, "")
                .Replace(@"\", "").Replace("\"", "");

            if (StopWordsList.Contains(normalizedToken)) continue;
            
            if (tfDict.TryGetValue(normalizedToken, out var value) && DfDict.ContainsKey(normalizedToken)) 
                tfDict[normalizedToken] = ++value;
            
            else if(DfDict.ContainsKey(normalizedToken)) tfDict.TryAdd(normalizedToken, 1);
            
        }
        //compute tfidf
        var vector = new List<double>(DfDict.Count);
        foreach (var key in DfDict.Keys)
        {
            if (!tfDict.TryGetValue(key, out int value)) vector.Add(0);
            else vector.Add(value * _idfDict[key]);
        }

        //add bias term
        vector.Insert(0, 1);
        var feature = Vector<double>.Build.DenseOfEnumerable(vector);
        return feature;
    }

    public static List<NaiveBayes.Document> CreateTrainCorpusFromXlsx(string filePath)
    {
        var trainCorpus = new List<NaiveBayes.Document> { new ("1", "2") };
        var wb = new Workbook(filePath);

        var worksheet = wb.Worksheets[0];

        var rows = worksheet.Cells.MaxDataRow;

        for (var i = 0; i < rows; i++)
        {
            trainCorpus.Add(new NaiveBayes.Document(worksheet.Cells[i, 0].Value.ToString(), worksheet.Cells[i, 1].Value.ToString()));
        }

        return trainCorpus;
    }
    public static Matrix<double> VectorizeFeaturesOneVsAll(Frame<int, string> df)
    {
        Console.WriteLine("Vectorizing traing data...");
        var header = df.GetColumn<string>("Header").Values.ToList();
        var description = df.GetColumn<string>("Description").Values.ToList();
        var concatedFeatures = new string[header.Count];
        for (var i = 0; i < header.Count; i++)
        {
            concatedFeatures[i] = header[i] + " " + description[i];
        }

        _numOfDocs = concatedFeatures.Length;
        var tfDicts = concatedFeatures.Select(TokenizeAndTf).ToList();
        //compute idf
        _idfDict = ComputeIdf();
        //compute tfidf
        var features = Matrix<double>.Build.Dense(_numOfDocs, DfDict.Count);
        var rowindex = 0;
        foreach (var tfdict in tfDicts)
        {
            var tfidfVector = new List<double>(DfDict.Count);
            foreach (var key in DfDict.Keys)
            {
                if (!tfdict.TryGetValue(key, out var value))
                {
                    tfidfVector.Add(0);
                }
                else
                {
                    tfidfVector.Add(value * _idfDict[key]);
                }
            }
            Vector<double> vec = Vector<double>.Build.DenseOfEnumerable(tfidfVector);
            features.SetRow(rowindex, vec);
            rowindex++;
        }
        _featureMatrix = features;
        //add bias term
        var ones = new double[features.RowCount];
        Array.Fill(ones, 1.0);
        var bias = Vector<double>.Build.DenseOfArray(ones);
        var featureMatrix = features.InsertColumn(0, bias);
        Console.WriteLine("Done!");
        return featureMatrix;
    }

    public static double[][] GetFeaturesForKMeans(Frame<int, string> df)
    {
        var samples = df.GetColumn<string>("Text").Values.ToArray();

        var wordFrequencyDictionary = new Dictionary<string, int>();

        foreach (var item in samples)
        {
            var tokens = item.Split(" ");

            foreach (string token in tokens)
            {
                var loweredToken = token.ToLower();
                var normalizedToken = MyRegex().Replace(loweredToken, "")
                    .Replace(@"\", "").Replace("\"", "");

                if (StopWordsList.Contains(normalizedToken)) continue;
                if (wordFrequencyDictionary.ContainsKey(normalizedToken))
                {
                    wordFrequencyDictionary[normalizedToken] += 1;
                }
                else
                {
                    wordFrequencyDictionary[normalizedToken] = 1;
                }
            }
        }
        wordFrequencyDictionary = (from entry in wordFrequencyDictionary orderby entry.Value descending select entry)
            .Where(pair => pair.Value > 9)
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        _kMeansWordFreqDict = wordFrequencyDictionary;
        var featureMatrix = Matrix<double>.Build.Dense(samples.Length, wordFrequencyDictionary.Count);

        int itemIndex = 0;
        foreach (var item in samples)
        {
            var localFrequency = new Dictionary<string, int>();
            var tokens = item.Split(" ");

            foreach (string token in tokens)
            {
                var loweredToken = token.ToLower();
                var normalizedToken = MyRegex().Replace(loweredToken, "")
                    .Replace(@"\", "").Replace("\"", "");

                if (StopWordsList.Contains(normalizedToken)) continue;
                if (!localFrequency.ContainsKey(normalizedToken))
                    localFrequency[normalizedToken] = 1;

            }
            double[] inputVector = new double[wordFrequencyDictionary.Count];

            foreach (KeyValuePair<string, int> pair in localFrequency)
            {
                if (wordFrequencyDictionary.ContainsKey(pair.Key))
                {
                    int index = wordFrequencyDictionary.Keys.ToList().IndexOf(pair.Key);
                    inputVector[index] = pair.Value;
                }
            }
            featureMatrix.SetRow(itemIndex, inputVector);
            itemIndex++;
        }
        _kMeansFeatureMatrix = featureMatrix;
        return featureMatrix.ToRowArrays();
    }
    public static double[] VectorizeOneFeatureKMeans(string input)
    {
        var localFrequency = new Dictionary<string, int>();
        var tokens = input.Split(" ");

        foreach (string token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = MyRegex().Replace(loweredToken, "")
                .Replace(@"\", "").Replace("\"", "");

            if (StopWordsList.Contains(normalizedToken)) continue;
            if (!localFrequency.ContainsKey(normalizedToken))
                localFrequency[normalizedToken] = 1;
        }
        double[] inputVector = new double[_kMeansWordFreqDict.Count];

        foreach (var pair in localFrequency)
        {
            if (_kMeansWordFreqDict.ContainsKey(pair.Key))
            {
                int index = _kMeansWordFreqDict.Keys.ToList().IndexOf(pair.Key);
                inputVector[index] = pair.Value;
            }
        }
        return inputVector;

    }
    [GeneratedRegex("[-.?!)(,:0123456789/\t\n$%^*}{#@<>'['~;@]")]
    private static partial Regex MyRegex();

}
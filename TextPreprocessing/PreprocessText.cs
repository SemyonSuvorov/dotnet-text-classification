using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using Deedle;


namespace TextPreprocessing;

public class PreprocessText
{
    //df for each token
    private int _numOfDocs;
    private Dictionary<string, double> _idfDict;
    private readonly Dictionary<string, int> _dfDict = new();
    private readonly List<string> _stopWordsList = new();
    public PreprocessText()
    {
        const string filePath = "stopwords.txt";
        var stream = new FileStream(filePath, FileMode.Open);
        var reader = new StreamReader(stream);
        var parts = reader.ReadLine()?.Split(',');

        if (parts != null)
            foreach (var t in parts)
            {
                _stopWordsList.Add(t);
            }

        reader.Close();
    }
    private Dictionary<string, int> TokenizeAndTf(string input)
    {
        var tokens = input.Split(" ").ToList();
        var tfDict = new Dictionary<string, int>();
        List<string> listOfTokens = new();
        
        foreach (var token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = Regex.Replace(loweredToken, 
                "[-.?!)(,:0123456789/\t\n$%^*}{#@<>'['~;@]", "")
                .Replace(@"\", "").Replace("\"","");
            if (_stopWordsList.Contains(normalizedToken))
            {
                continue;
            }
            if (tfDict.ContainsKey(normalizedToken))
            {
                tfDict[normalizedToken]++;
            }
            else
            {
                tfDict.TryAdd(normalizedToken, 1);
            }

            if (!listOfTokens.Contains(normalizedToken))
            {
                listOfTokens.Add(normalizedToken);
            }
            
        }
        foreach (var tok in listOfTokens)
        {
            if (_dfDict.ContainsKey(tok))
            {
                _dfDict[tok]++;
            }
            else
            {
                _dfDict.TryAdd(tok, 1);
            }
        }
        return tfDict;
    }

    private Dictionary<string, double> ComputeIdf()
    {
        return _dfDict.ToDictionary(pair => pair.Key, pair => Math.Log(_numOfDocs / pair.Value));
    }

    public Vector<double> VectorizeOneFeature(string input)
    {
        var tokens = input.Split(" ").ToList();
        var tfDict = new Dictionary<string, int>();

        foreach (var token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = Regex.Replace(loweredToken, 
                    "[-.?!)(,:0123456789/\t\n$%^*}{#@<>'['~;@]", "")
                .Replace(@"\", "").Replace("\"","");
            if (_stopWordsList.Contains(normalizedToken))
            {
                continue;
            }
            if (tfDict.ContainsKey(normalizedToken) && _dfDict.ContainsKey(normalizedToken))
            {
                tfDict[normalizedToken]++;
            }
            else if(_dfDict.ContainsKey(normalizedToken))
            {
                tfDict.TryAdd(normalizedToken, 1);
            }
        }
        //compute tfidf
        var vector = new List<double>(_dfDict.Count);
        foreach (var key in _dfDict.Keys)
        {
            if (!tfDict.ContainsKey(key))
            {
                vector.Add(0);
            }
            else
            {
                vector.Add(tfDict[key] * _idfDict[key]);
            }
        }
        //add bias term
        vector.Insert(0, 1);
        Vector<double> feature = Vector<double>.Build.DenseOfEnumerable(vector);
        return feature;
    }

    public Matrix<double> VectorizeFeatures(Frame<int, string> df)
    {
        Console.WriteLine("Vectorizing traing data...");
        var header = df.GetColumn<string>("Header").Values.ToList();
        var description = df.GetColumn<string>("Description").Values.ToList();
        var concatedFeatures = new string[header.Count];
        for (int i = 0; i < header.Count; i++)
        {
            concatedFeatures[i] = header[i] + " " + description[i];
        }
        
        List<Dictionary<string, int>> tfDicts = new();
        _numOfDocs = concatedFeatures.Length;
        foreach (var feature in concatedFeatures)
        {
            tfDicts.Add(TokenizeAndTf(feature));
        }
        //compute idf
        _idfDict = ComputeIdf();
        //compute tfidf
        Matrix<double> features = Matrix<double>.Build.Dense(_numOfDocs, _dfDict.Count);
        int rowindex = 0;
        foreach (var tfdict in tfDicts)
        {
            var tfidfVector = new List<double>(_dfDict.Count);
            foreach (var key in _dfDict.Keys)
            {
                if (!tfdict.ContainsKey(key))
                {
                    tfidfVector.Add(0);
                }
                else
                {
                    tfidfVector.Add(tfdict[key] * _idfDict[key]);
                }
            }
            Vector<double> vec = Vector<double>.Build.DenseOfEnumerable(tfidfVector);
            features.SetRow(rowindex, vec);
            rowindex++;
        }
        //add bias term
        var ones = new double[features.RowCount];
        Array.Fill(ones, 1.0);
        var bias = Vector<double>.Build.DenseOfArray(ones);
        var featureMatrix = features.InsertColumn(0, bias);
        Console.WriteLine("Done!");
        return featureMatrix;
    }
}
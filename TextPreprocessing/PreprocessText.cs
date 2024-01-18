using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using Aspose.Cells;
using Deedle;


namespace TextPreprocessing;

public partial class PreprocessText
{
    private int _numOfDocs;
    private Dictionary<string, double> _idfDict = new();
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
            var normalizedToken = MyRegex().Replace(loweredToken, "")
                .Replace(@"\", "").Replace("\"","");
            
            if (_stopWordsList.Contains(normalizedToken)) continue;
            
            if (tfDict.TryGetValue(normalizedToken, out int value)) tfDict[normalizedToken] = ++value;
            
            else tfDict.TryAdd(normalizedToken, 1);
            
            if (!listOfTokens.Contains(normalizedToken)) listOfTokens.Add(normalizedToken);
        }
        foreach (var tok in listOfTokens)
        {
            if (_dfDict.TryGetValue(tok, out int value)) _dfDict[tok] = ++value;
            
            else _dfDict.TryAdd(tok, 1);
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
        //tokenize 
        foreach (var token in tokens)
        {
            var loweredToken = token.ToLower();
            var normalizedToken = MyRegex().Replace(loweredToken, "")
                .Replace(@"\", "").Replace("\"", "");

            if (_stopWordsList.Contains(normalizedToken)) continue;
            
            if (tfDict.TryGetValue(normalizedToken, out int value) && _dfDict.ContainsKey(normalizedToken)) 
                tfDict[normalizedToken] = ++value;
            
            else if(_dfDict.ContainsKey(normalizedToken)) tfDict.TryAdd(normalizedToken, 1);
            
        }
        //compute tfidf
        var vector = new List<double>(_dfDict.Count);
        foreach (var key in _dfDict.Keys)
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

        int rows = worksheet.Cells.MaxDataRow;

        for (int i = 0; i < rows; i++)
        {
            trainCorpus.Add(new NaiveBayes.Document(worksheet.Cells[i, 0].Value.ToString(), worksheet.Cells[i, 1].Value.ToString()));
        }

        return trainCorpus;
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
                if (!tfdict.TryGetValue(key, out int value))
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
        //add bias term
        var ones = new double[features.RowCount];
        Array.Fill(ones, 1.0);
        var bias = Vector<double>.Build.DenseOfArray(ones);
        var featureMatrix = features.InsertColumn(0, bias);
        Console.WriteLine("Done!");
        return featureMatrix;
    }

    [GeneratedRegex("[-.?!)(,:0123456789/\t\n$%^*}{#@<>'['~;@]")]
    private static partial Regex MyRegex();

}
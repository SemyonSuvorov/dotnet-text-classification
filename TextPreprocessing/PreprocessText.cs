using System.Text.RegularExpressions;
using Deedle;


namespace TextPreprocessing;

public class PreprocessText
{
    private Dictionary<string, int> _dictOfWords = new ();

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
    
    private void TokenizeIntoWords(string input)
    {
        var tokens = input.Split(" ").ToList();
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
            
            if (_dictOfWords.ContainsKey(normalizedToken))
            {
                _dictOfWords[normalizedToken]++;
            }
            
            _dictOfWords.TryAdd(normalizedToken, 1);
        }
    }

    public void TokenizeTextCorpus(Frame<int, string> df)
    {
        var header = df.GetColumn<string>("Header").Values.ToList();
        var description = df.GetColumn<string>("Description").Values.ToList();
        var concatedFeature = new string[header.Count];
        for (int i = 0; i < header.Count; i++)
        {
            concatedFeature[i] = header[i] + " " + description[i];
        }

        foreach (var feature in concatedFeature)
        {
            TokenizeIntoWords(feature);
        }
    }

    public void PrintDict()
    {
        foreach (var pair in _dictOfWords.OrderByDescending(pair => pair.Value))
        {
            Console.WriteLine($"{pair.Key} -> {pair.Value}");
        }
    }
}
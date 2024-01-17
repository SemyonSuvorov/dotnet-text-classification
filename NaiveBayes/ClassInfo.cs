namespace NaiveBayes;

public class ClassInfo
{
    public int Label { get; }
    public int WordsCount { get; }
    private Dictionary<string, int> WordCount { get; }
    public int NumberOfDocs { get; }
    public ClassInfo(int label, List<String> trainDocs)
    {
        Label = label;
        var features = trainDocs.SelectMany(x => x.ExtractFeatures());
        WordsCount = features.Count();
        WordCount = features.GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
        NumberOfDocs = trainDocs.Count;
    }

    public int NumberOfOccurencesInTrainDocs(string word)
    {
        return WordCount.TryGetValue(word, out var value) ? value : 0;
    }
}
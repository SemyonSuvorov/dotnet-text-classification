namespace NaiveBayes;

public class NaiveBayesClassifier
{
    private readonly List<ClassInfo> _classes;
    private readonly int _countOfDocs;
    private readonly int _uniqWordsCount;
    
    public NaiveBayesClassifier(List<Document> train)
    {
        Console.WriteLine("Training Naive Bayes...");
        _classes = train.GroupBy(x => x.Label).Select(g => 
            new ClassInfo(Convert.ToInt32(g.Key), g.Select(x => x.Text).ToList())).ToList();
        _countOfDocs = train.Count;
        _uniqWordsCount = train.SelectMany(x => x.Text.Split(' ')).GroupBy(x => x).Count();
        Console.WriteLine("Done, ready to make predictions...");
    }
    public double PredictClass(string input)
    {
        var probas = new double[4];
        var predictedClass = 0;
        var maxProba = double.MinValue;
        for (var i = 0; i < 4; i++)
        {
            probas[i] = IsInClassProbability(i+1, input);
            if (!(probas[i] > maxProba)) continue;
            maxProba = probas[i];
            predictedClass = i + 1;
        }
        return predictedClass;
    }
    private double IsInClassProbability(int className, string text)
    {
        var words = text.ExtractFeatures();
        var classResults = _classes
            .Select(x => new
            {
                Result = Math.Pow(Math.E, Calc(x.NumberOfDocs, _countOfDocs, words, x.WordsCount, x, _uniqWordsCount)),
                ClassName = x.Label
            });
        
        return classResults.Single(x => x.ClassName == className).Result / classResults.Sum(x => x.Result);
    }

    private static double Calc(double dc, double d, List<string> q, double lc, ClassInfo @class, double v)
    {
        return Math.Log(dc / d) + q.Sum(x => Math.Log((@class.NumberOfOccurencesInTrainDocs(x) + 1) / (v + lc)));
    }
}
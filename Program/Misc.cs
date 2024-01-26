using ClassLibrary;
using OneVsAll;
using NaiveBayes;
using KMeans;

namespace LogReg;

public static class Misc
{
    public static int Menu()
    {
        Console.WriteLine("1. Make news headlines predictions with OneVsAll Classifier");
        Console.WriteLine("2. Make news headlines predictions with Naive Bayes Classifier");
        Console.WriteLine("3. Make Wiki articles predictions with KMeans (NBA/Company/Music)");
        Console.WriteLine("0. Exit");
        var s = Console.ReadLine();
        var c = int.TryParse(s, out var a);
        return !c ? 0 : a;
    }

    public static void PredictKMeans(KMeans.KMeans kmeans)
    {
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with KMeans. Please, type an article (Type Q to Quit)");
            var input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q" || string.IsNullOrWhiteSpace(input)) break;
            Console.WriteLine();
            // Get a prediction
            var result = kmeans.ClassifyString(input);
            // Print classification
            Console.WriteLine($"Predicted cluster: {(ClusterIntents)result}");
            Console.WriteLine();
        }
        while (true);
    }

    public static void PredictOneVsAll(OneVsAllClassifier oneVsAllClassifier)
    {
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with One Vs All. Please, enter news headline (Type Q to Quit)");
            var input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q" || string.IsNullOrWhiteSpace(input)) break;
            // Get a prediction
            var result = oneVsAllClassifier.PredictClassForSample(input);
            // Print classification
            Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
            Console.WriteLine();
        }
        while (true);
    }

    public static void PredictNaiveBayes(NaiveBayesClassifier naiveBayesClassifier)
    {
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with Naive Bayes. Please, enter news headline (Type Q to Quit)");
            var input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q" || string.IsNullOrWhiteSpace(input)) break;            
            // Get a prediction
            var result = naiveBayesClassifier.PredictClass(input);
            // Print classification
            Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
            Console.WriteLine();
        }
        while (true);
    }
}
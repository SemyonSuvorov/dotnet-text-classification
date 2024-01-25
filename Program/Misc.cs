using ClassLibrary;
using OneVsAll;
using NaiveBayes;

namespace LogReg;

public static class Misc
{
    public static int Menu()
    {
        Console.WriteLine("1. Make predictions with OneVsAll Classifier");
        Console.WriteLine("2. Make predictions with Naive Bayes Classifier");
        Console.WriteLine("3. Make predictions with KMeans");
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
            Console.WriteLine("Predicting with KMeans. What do you want to say ? (Type Q to Quit)");
            var input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q" || string.IsNullOrWhiteSpace(input)) break;
            // Get a prediction
            var result = kmeans.ClassifyString(input);
            // Print classification
            Console.WriteLine($"Predicted cluster: {result}");
            Console.WriteLine();
        }
        while (true);
    }

    public static void PredictOneVsAll(OneVsAllClassifier oneVsAllClassifier)
    {
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with One Vs All. What do you want to say ? (Type Q to Quit)");
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
            Console.WriteLine("Predicting with Naive Bayes. What do you want to say ? (Type Q to Quit)");
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
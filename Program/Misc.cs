using ClassLibrary;
using NaiveBayes;

namespace LogReg;

public class Misc
{
    public static int Menu()
    {
        int a;
        Console.WriteLine("1. Make predictions with OneVsAll Classifier");
        Console.WriteLine("2. Make predictions with Naive Bayes Classifier");
        Console.WriteLine("0. Exit");
        var s = Console.ReadLine();
        var c = int.TryParse(s, out a);
        if (!c) throw new Exception($"Failed to parse {s}");
        return a;
    }

    public static void PredictOneVsAll(OneVsAllClassifier oneVsAllClassifier)
    {
        string input;
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with One Vs All. What do you want to say ? (Type Q to Quit)");
            input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q") break;
            // Get a prediction
            var result = oneVsAllClassifier.PredictClassForSample(input);
            // Print classification
            Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
            Console.WriteLine();
        }
        while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");
    }

    public static void PredictNaiveBayes(NaiveBayesClassifier naiveBayesClassifier)
    {
        string? input;
        do
        {
            Console.WriteLine();
            Console.WriteLine("Predicting with Naive Bayes. What do you want to say ? (Type Q to Quit)");
            input = Console.ReadLine()!;
            if (input.ToLowerInvariant() == "q") break;
            // Get a prediction
            var result = naiveBayesClassifier.PredictClass(input);
            // Print classification
            Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
            Console.WriteLine();
        }
        while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");
    }
}
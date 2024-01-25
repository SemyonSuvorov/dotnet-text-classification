using System.Text.RegularExpressions;

namespace NaiveBayes;

public static partial class Helpers
{
    public static List<string> ExtractFeatures(this string text)
    {
        return MyRegex().Replace(text, "").Split(' ').ToList();
    }

    [GeneratedRegex("\\p{P}+")]
    private static partial Regex MyRegex();
}
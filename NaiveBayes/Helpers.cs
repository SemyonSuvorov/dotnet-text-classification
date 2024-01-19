using System.Text.RegularExpressions;

namespace NaiveBayes;

public static class Helpers
{
    public static List<string> ExtractFeatures(this string text)
    {
        return Regex.Replace(text, "\\p{P}+", "").Split(' ').ToList();
    }

}
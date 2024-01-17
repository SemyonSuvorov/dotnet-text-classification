namespace NaiveBayes;

public class Document
{
    public Document(string label, string text)
    {
        Label = label;
        Text = text;
    }
    public string Label { get; }
    public string Text { get; }
}
using ClassLibrary;
using NaiveBayes;
using Aspose.Cells;
using Deedle;
using TextPreprocessing;
using Frame = Deedle.Frame;

/*
var frame = Frame.ReadCsv("test.csv");
var trainRowIndices = new int[7000];
for (int i = 0; i < 7000; i++)
{
    trainRowIndices[i] = i;
}
var trainFrame = frame.GetRows(trainRowIndices);
*/
var p = new PreprocessText();
var trainCorpus = p.CreateTrainCorpusFromXlsx("TrainData.xlsx");
var c = new Classifier(trainCorpus);
string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What do you want to say ? (Type Q to Quit)");
    input = Console.ReadLine()!;
    if (input.ToLowerInvariant() == "q") break;
    // Get a prediction
    var result = c.PredictClass(input);
    // Print classification
    Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");
Console.WriteLine("Bye!");
/*
var classifier = new OneVsAllClassifier(trainFrame, 100);
classifier.Train(11);
string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What do you want to say ? (Type Q to Quit)");
    input = Console.ReadLine()!;
    if (input.ToLowerInvariant() == "q") break;
    // Get a prediction
    var result = classifier.PredictClassForSample(input);
    // Print classification
    Console.WriteLine($"Predicted class: {(ArticleIntents)result}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");
Console.WriteLine("Bye!");
*/
                                                  
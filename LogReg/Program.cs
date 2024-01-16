using ClassLibrary;
using Deedle;
using TextPreprocessing;
using Frame = Deedle.Frame;


/*
Frame<int, string> frame = Frame.ReadCsv("simple_data_for_linear_regression.csv");
var trainRowIndices = new int[300];
for (int i = 0; i < 300; i++)
{
    trainRowIndices[i] = i;
}

var trainFrame = frame.GetRows(trainRowIndices);
var testRowIndices = new int[100];
int j = 0;
for (int i = 300; i < 400; i++)
{
    testRowIndices[j] = i;
    j++;
}

var testFrame = frame.GetRows(testRowIndices);
var yTrue = testFrame.GetColumn<string>("Target").Select(x => x.Value).Values.ToList();
var logreg = new LogisticRegression(10000);
logreg.Fit(trainFrame, 0.01);

var preds = logreg.PredictProbaForMultipleSamples(testFrame);
for(int i = 0; i < preds.Count; i++)
{
    Console.WriteLine($"Predicted proba: {preds[i]} -> Real label: {yTrue[i]}");
}

var oneSample = new List<double>()
{
    365.0, -1.0978727092313199, 0.5766048932140662,
    -0.0030922379575568136, 0.5469982296454587, -1.0001157199466832, 0.2471083764924692,
    -0.4616078693750237, -0.028045418922321503, 0.5310816568918002
}; //1
var predForOneSample = logreg.PredictProbaForOneSample(oneSample);
Console.WriteLine(predForOneSample);
*/


var p = new PreprocessText();

var toyFrame = Frame.ReadCsv("test.csv");
p.TokenizeTextCorpus(toyFrame);
p.PrintDict();
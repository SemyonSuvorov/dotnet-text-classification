using ClassLibrary;
using Deedle;

using Frame = Deedle.Frame;



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

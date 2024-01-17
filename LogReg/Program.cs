using ClassLibrary;
using Frame = Deedle.Frame;

var toyFrame = Frame.ReadCsv("test.csv");
var classifier = new OneVsAllClassifier(toyFrame, 100);
classifier.Train(11);
var result = classifier.PredictClassForSample("Indians Mount Charge,The Cleveland Indians pulled within one " +
                                              "game of the AL Central lead by beating the Minnesota Twins, " +
                                              "7-1, Saturday night with home runs by Travis Hafner and " +
                                              "Victor Martinez.");
foreach (var r in result)
{
    Console.WriteLine(r);
} // 2
                                                  
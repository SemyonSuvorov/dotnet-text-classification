using ClassLibrary;
using Frame = Deedle.Frame;

var logreg = new LogisticRegression(100);
var toyFrame = Frame.ReadCsv("test.csv");
logreg.Fit(toyFrame, 11);
Console.WriteLine(logreg.PredictProbaForOneSample("Indians Mount Charge,The Cleveland Indians pulled within one " +
                                                  "game of the AL Central lead by beating the Minnesota Twins, " +
                                                  "7-1, Saturday night with home runs by Travis Hafner and " +
                                                  "Victor Martinez.")); // 2
                                                  
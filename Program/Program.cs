using ClassLibrary;
using NaiveBayes;
using Deedle;
using LogReg;
using TextPreprocessing;
using Frame = Deedle.Frame;


//load text for onevsall
var p = new PreprocessText();
var frame = Frame.ReadCsv("test.csv");
var trainRowIndices = new int[7000];
for (int i = 0; i < 7000; i++)
{
    trainRowIndices[i] = i;
}
var trainFrame = frame.GetRows(trainRowIndices);


var oneVsAllClassifier = new OneVsAllClassifier(trainFrame, 100);
oneVsAllClassifier.Train(11);

//load text for naive bayes
var trainCorpus = p.CreateTrainCorpusFromXlsx("TrainData.xlsx");
var naiveBayesClassifier = new NaiveBayesClassifier(trainCorpus);

while (true)
{
    switch (Misc.Menu())
    {
        case 1:
            Misc.PredictOneVsAll(oneVsAllClassifier);
            Console.WriteLine("Finished!");
            break;
        case 2:
            Misc.PredictNaiveBayes(naiveBayesClassifier);
            Console.WriteLine("Finished!");
            break;
        default:
            Console.WriteLine("Bye!");
            return 0;
    }
}


                                                  
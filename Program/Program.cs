using ClassLibrary;
using OneVsAll;
using NaiveBayes;
using Deedle;
using LogReg;
using TextPreprocessing;
using Frame = Deedle.Frame;


//load text for one vs all
var frame = Frame.ReadCsv("test.csv");
var trainRowIndices = new int[7000];
for (var i = 0; i < 7000; i++)
{
    trainRowIndices[i] = i;
}
var trainFrame = frame.GetRows(trainRowIndices);


var oneVsAllClassifier = new OneVsAllClassifier(trainFrame, 100);
oneVsAllClassifier.Train(12);

var kmeans = new KMeans.KMeans(trainFrame, 4);

//load text for naive bayes
var trainCorpus = PreprocessText.CreateTrainCorpusFromXlsx("TrainData.xlsx");
var naiveBayesClassifier = new NaiveBayesClassifier(trainCorpus);

while (true)
{
    switch (Misc.Menu())
    {
        case 1:
            Misc.PredictOneVsAll(oneVsAllClassifier);
            Console.WriteLine("Finished!");
            Console.WriteLine();
            break;
        case 2:
            Misc.PredictNaiveBayes(naiveBayesClassifier);
            Console.WriteLine("Finished!");
            Console.WriteLine();
            break;
        case 3:
            Misc.PredictKMeans(kmeans);
            Console.WriteLine("Finished!");
            Console.WriteLine();
            break;
        default:
            Console.WriteLine("Bye!");
            return 0;
    }
}


                                                  
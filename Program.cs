using System;
using System.IO;
using Microsoft.ML;

namespace Taxi_fare_ML
{
    class Program
    {
        //_trainDataPath contains the path to the file with the data set used to train the model.
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");

        //_testDataPath contains the path to the file with the data set used to evaluate the model.
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

        //_modelPath contains the path to the file where the trained model is stored.
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }
}

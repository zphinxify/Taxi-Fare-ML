﻿using System;
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
            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return (model);
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            //The Transform() method makes predictions for the test dataset input rows.
            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");


            Console.WriteLine();
            Console.WriteLine($"*===============================================*");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*===============================================*");
            Console.WriteLine();

            // RSquared takes values between 0 and 1. The closer its value is to 1, the better the model is
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

            // RMS stems from the regression model. The lower the value, the better the model
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }


        // Predicts fare amount based on test data, displays the predicted results
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            // Uses the CreatePredictionEngine to predict the Taxifare
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            //Sample data to test the prediction-engine
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0
            };

            // Run prediction pipeline on one example
            var prediction = predictionFunction.Predict(taxiTripSample);


            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

    }
}

# Taxi-Fare-ML

The provided data set contains the following columns:

vendor_id: The ID of the taxi vendor is a feature.

rate_code: The rate type of the taxi trip is a feature.

passenger_count: The number of passengers on the trip is a feature.

trip_time_in_secs: The amount of time the trip took. You want to predict the fare of the trip before the trip is completed. At that moment, you don't know how long the trip would take. Thus, the trip time is not a feature and you'll exclude this column from the model.

trip_distance: The distance of the trip is a feature.

payment_type: The payment method (cash or credit card) is a feature.

fare_amount: The total taxi fare paid is the label.
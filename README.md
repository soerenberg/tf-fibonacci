# tf-fibonacci

[![Build Status](https://travis-ci.com/soerenberg/tf-fibonacci.svg?branch=main)](https://travis-ci.com/soerenberg/tf-fibonacci)

This is a simple example of computing Fibonacci in TensorFlow, exporting the
computation graph as a TensorFlow SavedModel and finally serve the resulting
SavedModel using TensorFlow-Serving using Docker.

## Serving the model using TF serving and docker

Run or follow the steps in `serve_model.sh` to deploy the model on your
localhost.

You can then send http requests to make predictions as follows

```
curl -X POST \
  -H "Content-type: application/json" \
  -H "Accept: application/json" \
  -d '{"instances": [0, 25, 5, 10, 50]}' \
  "http://localhost:8501/v1/models/fibonacci:predict"

>>{
    "predictions": [0, 75025, 5, 55, 12586269025]
}
```

Alternatively, send gRPC requests to port `8500`.

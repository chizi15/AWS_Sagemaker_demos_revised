import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import datetime
import boto3
import s3fs
import sagemaker
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

# from sagemaker import get_execution_role
np.random.seed(0)

# set aws parameters
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'
region_name = 'cn-northwest-1'
role = 'your_role'
sagemaker_session = sagemaker.Session(boto3.session.Session(profile_name='default'))
bucket = 'your_bucket'
prefix = 'sagemaker/zc/DEMO-deepar'
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)
image_name = sagemaker.image_uris.retrieve(framework='forecasting-deepar', region=region_name, version='1')

# set data and model parameters
freq = 'D'
prediction_length = 28
context_length = 28 * 2
t0 = '2020-01-01'
data_length = 365 * 3
num_ts = 4 * 1
period = 365

# generate training and testing data
time_series = []
for k in range(num_ts):
    level = 10 * np.random.rand()
    seas_amplitude = (0.1 + 0.3 * np.random.rand()) * level
    sig = 0.05 * level  # noise parameter (constant in time)
    time_ticks = np.array(range(data_length))
    source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
    noise = sig * np.random.randn(data_length)
    data = source + noise
    index = pd.date_range(t0, periods=data_length, freq='D')
    time_series.append(pd.Series(data=data, index=index))
time_series[num_ts - 1].plot()
plt.show()

time_series_training = []
for ts in time_series:
    time_series_training.append(ts[:-prediction_length])
time_series[num_ts - 1].plot(label='test')
time_series_training[num_ts - 1].plot(label='train', ls=':')
plt.legend()
plt.show()


# convert pd.series(time series) to dict, then to json line
def series_to_obj(ts, cat=None):
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj


def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))


# upload data to s3
encoding = "utf-8"
s3filesystem = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
with s3filesystem.open(s3_data_path + "/train/train-4.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))
with s3filesystem.open(s3_data_path + "/test/test-4.json", 'wb') as fp:
    for ts in time_series:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

training_type = 'normal_training'  # normal_training or hypers_tuning

# normal training, no hypers tuning
if training_type == 'normal_training':
    estimator = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=image_name,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        max_run=3600,
        use_spot_instances=True,
        max_wait=3600,
        # MaxWaitTimeInSeconds(i.e. max_wait) above 3600 is not supported for the given algorithm,
        # and It must be present and be greater than or equal to MaxRuntimeInSeconds(i.e. max_run).
        base_job_name='DEMO-deepar-synthetic-zc',  # 不能用下划线_
        output_path="s3://" + s3_output_path,
        tags=[{'Key': 'app', 'Value': 'sagemaker-deepar-demo-test'},
              {'Key': 'env', 'Value': 'test'},
              {'Key': 'name', 'Value': 'synthetic-data'},
              {'Key': 'depart', 'Value': 'rd7'},
              {'Key': 'manage', 'Value': 'zhangchi'},
              {'Key': 'input data', 'Value': 'train-4.json'},
              {'Key': 'instance type / workers / parallel / count',
               'Value': 'ml.m5.xlarge / 4 / 1 / 1'}])  # tags中不能出现中英文逗号
    hyperparameters = {
        "time_freq": freq,
        "context_length": str(context_length),
        "prediction_length": str(prediction_length),
        "num_cells": "40",
        "num_layers": "2",
        "epochs": "100",
        "mini_batch_size": "32",
        "learning_rate": "0.001",
        "dropout_rate": "0.1",
        "early_stopping_patience": "10",
        'likelihood': 'gaussian'}
    estimator.set_hyperparameters(**hyperparameters)
    data_channels = {
        "train": "s3://{}/train/train-4.json".format(s3_data_path),
        "test": "s3://{}/test/test-4.json".format(s3_data_path)}
    estimator.fit(inputs=data_channels, wait=True)
    # deploy endpoint using training model
    job_name = estimator.latest_training_job.name
    endpoint_name = sagemaker_session.endpoint_from_job(
        job_name=job_name,
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        image_uri=image_name,
        role=role,
        wait=True,
        name=job_name)
    # name必须与job_name一致，否则会报"ValueError: Shape of passed values is (28, 9), indices imply (24, 9)"的错误，括号内数字可能不同。

# hypers tuning
elif training_type == 'hypers_tuning':
    estimator_hyper = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=image_name,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        max_run=3600,
        use_spot_instances=True,
        max_wait=3600,
        base_job_name='DEMO-deepar-synthetic-zc',
        output_path="s3://" + s3_output_path,
        hyperparameters={
            "time_freq": freq,
            "context_length": str(context_length),
            "prediction_length": str(prediction_length),
            "num_cells": "40",
            "num_layers": "2",
            "epochs": "100",
            "mini_batch_size": "32",
            "learning_rate": "0.001",
            "dropout_rate": "0.05",
            "early_stopping_patience": "10",
            'likelihood': 'gaussian'},
        tags=[{'Key': 'app', 'Value': 'sagemaker-deepar-demo-test'},
              {'Key': 'env', 'Value': 'test'},
              {'Key': 'name', 'Value': 'synthetic-data/hyperparameters-tuning'},
              {'Key': 'depart', 'Value': 'rd7'},
              {'Key': 'manage', 'Value': 'zhangchi'},
              {'Key': 'input data', 'Value': 'train-4.json'},
              {'Key': 'instance type / workers / parallel / count', 'Value': 'ml.m5.xlarge / 4 / 1 / 1'},
              {'Key': 'hypers choice', 'Value': 'likelihood: deterministic-L1 / student-T'}])  # tags中不能出现中英文逗号

    hyperparameter_ranges = {'likelihood': CategoricalParameter(["deterministic-L1", 'student-T'])}
    objective_metric_name = 'test:mean_wQuantileLoss'
    tuner = HyperparameterTuner(
        estimator_hyper,
        objective_metric_name,
        hyperparameter_ranges,
        strategy='Bayesian',
        objective_type='Minimize',
        max_jobs=1,
        max_parallel_jobs=1,
        tags=[{'Key': 'app', 'Value': 'sagemaker-deepar-demo-test'},
              {'Key': 'env', 'Value': 'test'},
              {'Key': 'name', 'Value': 'synthetic-data/hyperparameters-tuning'},
              {'Key': 'depart', 'Value': 'rd7'},
              {'Key': 'manage', 'Value': 'zhangchi'},
              {'Key': 'input data', 'Value': 'train-4.json'},
              {'Key': 'instance type / workers / parallel / count',
               'Value': 'ml.m5.xlarge / 4 / 1 / 1'},
              {'Key': 'hypers choice', 'Value': 'likelihood: deterministic-L1 / student-T'}])
    data_channels = {
        "train": "s3://{}/train/train-4.json".format(s3_data_path),
        "test": "s3://{}/test/test-4.json".format(s3_data_path)}
    tuner.fit(inputs=data_channels, wait=True)
    # tuner.fit(inputs=data_channels, wait=False)
    # boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    #                             HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)

    # deploy endpoint using training model
    job_name = tuner.best_training_job()  # 此处不能用tuner.latest_tuning_job.job_name，
    # 否则得到的不是job_name，而是Hyperparameter tuning jobs的name，会比job_name差几个最后的字符， 就会报"botocore.exceptions.ClientError: An
    # error occurred (ValidationException) when calling the DescribeTrainingJob operation: Requested resource not
    # found."的错误。
    endpoint_name = sagemaker_session.endpoint_from_job(
        job_name=job_name,
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        image_uri=image_name,
        role=role,
        wait=True,
        name=job_name)  # name 必须与 job_name一致，否则会报"ValueError: Shape of passed values is (28, 9), indices imply (24,
    # 9)"的错误，括号内数字可能不同。

else:
    print('\'training_type\' must be one of \'normal_training\' or \'hypers_tuning\'')


# define predict function
class DeepARPredictor(sagemaker.predictor.Predictor):

    def __init__(self, endpoint_name, sagemaker_session=None):
        super(DeepARPredictor, self).__init__(endpoint_name, sagemaker_session)
        self.serializer.content_type = 'application/json'

    def set_prediction_parameters(self, freq, prediction_length):
        """
        Set the time frequency and prediction length parameters. This method **must** be called
        before being able to use `predict`.
        Parameters:
        freq -- string indicating the time frequency
        prediction_length -- integer, number of predicted time points
        Return value: none.
        """
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100,
                quantiles=["0.1", '0.2', '0.3', '0.4', "0.5", '0.6', '0.7', '0.8', "0.9"]):
        """
        Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.
        Parameters:
        ts -- list of `pandas.Series` objects, the time series to predict
        cat -- list of integers (default: None)
        encoding -- string, encoding to use for the request (default: "utf-8")
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])
        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_times = [x.index[-1] + datetime.timedelta(days=1) for x in ts]  # set prediction starting time
        req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, prediction_times, encoding)

    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        instances = [series_to_obj(ts[k], cat[k] if cat else None) for k in range(len(ts))]
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)

    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range(len(prediction_times)):
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df


predictor = DeepARPredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session)
predictor.set_prediction_parameters(freq, prediction_length)
# predictor.serializer.content_type = "application/json"

# predict data
list_of_df = predictor.predict(time_series_training[-4:])
actual_data = time_series[-4:]
for k in range(len(list_of_df)):
    plt.figure(figsize=(12, 6))
    actual_data[k][-prediction_length - context_length:].plot(label='target')
    p10 = list_of_df[k]['0.1']
    p90 = list_of_df[k]['0.9']
    plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
    list_of_df[k]['0.5'].plot(label='prediction median')
    plt.legend()
    plt.show()

# delete endpoint machine
sagemaker_session.delete_endpoint(endpoint_name)

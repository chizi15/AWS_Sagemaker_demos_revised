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

# set aws parameters
aws_access_key_id = 'AKIA3ZMS56MGH7FMNUEF'
aws_secret_access_key = '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'
region_name = 'cn-northwest-1'
role = 'arn:aws-cn:iam::810446353164:role/service-role/AmazonSageMaker-ExecutionRole-20200814T152597'
sagemaker_session = sagemaker.Session(boto3.session.Session(
    aws_access_key_id = 'AKIA3ZMS56MGH7FMNUEF', aws_secret_access_key = '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'))
# sagemaker_session = sagemaker.Session(boto3.session.Session(
#     aws_access_key_id = 'AKIA3ZMS56MGH7FMNUEF', aws_secret_access_key = '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',
#     region_name = 'cn-northwest-1'))
# sagemaker_session = sagemaker.Session(boto3.session.Session(region_name = 'cn-northwest-1', profile_name='default'))
# sagemaker_session = sagemaker.Session()
bucket = 'sagemaker-cn-northwest-1-810446353164'
prefix = 'sagemaker/zc/DEMO-deepar'
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)
image_name = sagemaker.image_uris.retrieve(framework='forecasting-deepar', region=region_name, version='latest')


# set data and model parameters
freq = '1W'
mini_batch_size = 100
period = 52
prediction_length = int(period/4)  # The value for hyperparameter 'prediction_length' should be of type integer
context_length = prediction_length*1  # The value for hyperparameter 'context_length' should be of type integer
t0 = '2018-01-01'
data_length = period*3
num_ts = mini_batch_size * 1

np.random.seed(0)
plt.figure(figsize=(20,10))
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
    index = pd.date_range(t0, periods=data_length, freq=freq)
    time_series.append(pd.Series(data=data, index=index))

# 使time_series_training和time_series一一对应
time_series_training = []
for ts in time_series:
    time_series_training.append(ts[:-prediction_length])

# # 使time_series_training和time_series无关
# time_series_training = []
# for k in range(num_ts):
#     level = 10 * np.random.rand()
#     seas_amplitude = (0.1 + 0.3 * np.random.rand()) * level
#     sig = 0.05 * level  # noise parameter (constant in time)
#     time_ticks = np.array(range(data_length-prediction_length))
#     source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi) / period)
#     noise = sig * np.random.randn(data_length-prediction_length)
#     data = source + noise
#     index = pd.date_range(t0, periods=data_length-prediction_length, freq='D')
#     time_series_training.append(pd.Series(data=data, index=index))

num_of_samples = 10
for k in range(num_of_samples):
    time_series[num_ts - num_of_samples:][k].plot(label='actual')  # 只画出最后4条实际值序列，和最后与预测序列对比的实际值序列一一对应
    time_series_training[num_ts - num_of_samples:][k].plot(label='train', ls=':')  # 只画出最后4条训练序列，和最后的预测序列一一对应
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
with s3filesystem.open(s3_data_path + "/train/train-400.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))
with s3filesystem.open(s3_data_path + "/test/test-400.json", 'wb') as fp:
    for ts in time_series:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

training_type = 'hypers_tuning'  # normal_training or hypers_tuning

# normal training, no hypers tuning
if training_type == 'normal_training':
    estimator = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=image_name,
        role=role,
        instance_count=1,
        instance_type='ml.c5.xlarge',
        max_run=3600,
        use_spot_instances=True,
        max_wait=3600,
        # MaxWaitTimeInSeconds(i.e. max_wait) above 3600 is not supported for the given algorithm,
        # and It must be present and be greater than or equal to MaxRuntimeInSeconds(i.e. max_run).
        checkpoint_s3_uri='s3://sagemaker-cn-northwest-1-810446353164/DeepAR/HPO/model/checkpoint_s3_uri',
        checkpoint_local_path='/opt/ml/checkpoints',
        base_job_name='DEMO-deepar-synthetic-zc',  # 不能用下划线_
        output_path="s3://" + s3_output_path,
        tags=[{'Key': 'App', 'Value': 'Sagemaker-DeepAR'},
              {'Key': 'Depart', 'Value': 'RD7'},
              {'Key': 'Env', 'Value': 'Test'},
              {'Key': 'Manager', 'Value': 'Zhang Chi'},
              {'Key': 'Name', 'Value': 'DeepAR-Training'},
              {'Key': 'input data', 'Value': 'train-400.json'},
              {'Key': 'instance type / workers / parallel / count',
               'Value': 'ml.c5.xlarge / 4 / 1 / 1'}])  # tags中不能出现中英文逗号
    hyperparameters = {
        'num_eval_samples': prediction_length,
        'test_quantiles': [0.5, 0.6, 0.7, 0.8],
        # '_tuning_objective_metric': 'test:mean_wQuantileLoss',
        "time_freq": freq,
        "context_length": str(context_length),
        "prediction_length": str(prediction_length),  # prediction_length in predict and training must be the same
        "num_cells": "40",
        "num_layers": "2",
        'embedding_dimension': 10,
        "epochs": "50",
        "mini_batch_size": "32",
        "learning_rate": "0.001",
        "dropout_rate": "0.1",
        "early_stopping_patience": "10",
        'likelihood': 'gaussian'}
    estimator.set_hyperparameters(**hyperparameters)
    data_channels = {
        "train": "s3://{}/train/train-400.json".format(s3_data_path),}
        # "test": "s3://{}/test/test-400.json".format(s3_data_path)}
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
    # name必须与job_name一致，否则当不同模型的输入数据维度不一致时会报类似于"ValueError: Shape of passed values is (28, 9), indices imply (24, 9)"的错误，括号内数字可能不同。

# hypers tuning
elif training_type == 'hypers_tuning':
    estimator_hyper = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=image_name,
        role=role,
        instance_count=2,
        instance_type='ml.c5.xlarge',
        max_run=3600,
        use_spot_instances=True,
        max_wait=3600,
        base_job_name='DEMO-deepar-synthetic-zc',
        output_path="s3://" + s3_output_path,
        hyperparameters={
            'num_eval_samples': prediction_length,
            'test_quantiles': [0.5, 0.6, 0.7, 0.8],
            # 'tuning_objective_metric': 'test:mean_wQuantileLoss',
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
        tags=[{'Key': 'App', 'Value': 'Sagemaker-DeepAR'},
              {'Key': 'Depart', 'Value': 'RD7'},
              {'Key': 'Env', 'Value': 'Test'},
              {'Key': 'Manager', 'Value': 'Zhang Chi'},
              {'Key': 'Name', 'Value': 'DeepAR-Training'},
              {'Key': 'input data', 'Value': 'train-400.json'},
              {'Key': 'instance type / workers / parallel / count', 'Value': 'ml.m5.xlarge / 4 / 1 / 1'},
              {'Key': 'hypers choice', 'Value': 'likelihood: deterministic-L1 / student-T'}])  # tags中不能出现中英文逗号

    # hyperparameter_ranges = {'likelihood': CategoricalParameter(["deterministic-L1", 'student-T'])}
    hyperparameter_ranges = {'embedding_dimension': IntegerParameter(32, 64), 'num_cells': IntegerParameter(32, 64),
                             'num_layers': IntegerParameter(2, 3)}
    objective_metric_name = 'test:mean_wQuantileLoss'
    tuner = HyperparameterTuner(
        estimator_hyper,
        objective_metric_name,
        hyperparameter_ranges,
        strategy='Bayesian',
        objective_type='Minimize',
        max_jobs=2,
        max_parallel_jobs=1,
        tags=[{'Key': 'App', 'Value': 'Sagemaker-DeepAR'},
              {'Key': 'Depart', 'Value': 'RD7'},
              {'Key': 'Env', 'Value': 'Test'},
              {'Key': 'Manager', 'Value': 'Zhang Chi'},
              {'Key': 'Name', 'Value': 'DeepAR-HypersTuning'},
              {'Key': 'input data', 'Value': 'train-400.json'},
              {'Key': 'instance type / workers / parallel / count',
               'Value': 'ml.m5.xlarge / 4 / 1 / 1'},
              {'Key': 'hypers choice', 'Value': 'likelihood: deterministic-L1 / student-T'}])
    data_channels = {
        "train": "s3://{}/train/train-400.json".format(s3_data_path),
        "test": "s3://{}/test/test-400.json".format(s3_data_path)}
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
        self.prediction_length = prediction_length  # prediction_length in predict and training must be the same

    def predict(self, ts, cat=None, encoding="utf-8", num_samples=1000,
                quantiles=["0.1", '0.2', '0.3', '0.4', "0.5", '0.6', '0.7', '0.8', "0.9", '0.95']):
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


predictor = DeepARPredictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
predictor.set_prediction_parameters(freq, prediction_length)
# predictor.serializer.content_type = "application/json"

# predict data
list_of_df = predictor.predict(time_series_training[-num_of_samples:])
actual_data = time_series[-num_of_samples:]
for k in range(len(list_of_df)):
    plt.figure(figsize=(12, 6))
    # actual_data[k][-prediction_length - context_length:].plot(label='target')
    actual_data[k][-prediction_length:].plot(label='target')
    p10 = list_of_df[k]['0.1']
    p90 = list_of_df[k]['0.9']
    plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
    list_of_df[k]['0.5'].plot(label='prediction median')
    list_of_df[k]['0.95'].plot(label='prediction P95')
    time_series_training[-num_of_samples:][k].plot(label='input')
    plt.legend()
    plt.show()

# delete endpoint machine
sagemaker_session.delete_endpoint(endpoint_name)


"""deepar train: [ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.p4d.24xlarge, ml.c5n.xlarge, ml.p3.16xlarge, ml.m5.large, 
ml.p2.16xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.c5n.18xlarge, ml.g4dn.xlarge, 
ml.g4dn.12xlarge, ml.c4.8xlarge, ml.g4dn.2xlarge, ml.c5.9xlarge, ml.g4dn.4xlarge, ml.c5.xlarge, ml.g4dn.16xlarge, 
ml.c4.xlarge, ml.g4dn.8xlarge, ml.c5n.2xlarge, ml.c5n.4xlarge, ml.c5.18xlarge, ml.p3dn.24xlarge, ml.p3.2xlarge, 
ml.m5.xlarge, ml.m4.10xlarge, ml.c5n.9xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, 
ml.p2.8xlarge, ml.m5.2xlarge, ml.p3.8xlarge, ml.m4.4xlarge] """


"""
S3_BUCKET = "wuerp-sagemaker"
S3_PREDICT_PATH = "data/predict"
# estimator.latest_training_job.name
# 当训练和预测中断分离使，可手动对model_name赋值，拿到训练好的模型去预测
# 特别重要，model_name是sagemaker,inference,models里面的模型名称，不是training jobs里面的模型名称，二者在时间戳上有时会有细微差别
_transformer = sagemaker.transformer.Transformer(model_name='0042-top80test-week-2021-01-07-03-55-40-928',  # 特别重要，model_name是sagemaker,inference,models里面的模型名称，不是training jobs里面的模型名称，二者在时间戳上有时会有细微差别
                                                instance_count=1,
                                                instance_type="ml.c5.xlarge",
                                                strategy="MultiRecord",
                                                assemble_with="Line",
                                                output_path=f"s3://{S3_BUCKET}/{S3_PREDICT_PATH}/out_0138/",
                                                 tags=[{'Key': 'App', 'Value': 'Sagemaker-DeepAR'},
                                                       {'Key': 'Depart', 'Value': 'RD7'},
                                                       {'Key': 'Env', 'Value': 'Test'},
                                                       {'Key': 'Manager', 'Value': 'Zhang Chi'},
                                                       {'Key': 'Name', 'Value': 'DeepAR-Transform'}])

_transformer.transform(f"s3://{S3_BUCKET}/{S3_PREDICT_PATH}/in_0138/",
                          data_type="S3Prefix",
                          split_type="Line",
                          wait=False)

"""

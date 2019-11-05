#!/bin/bash

job_name=$1
if [[ -z ${job_name} ]]
then
    echo 'Provide model name'
    exit 0
fi
echo 'Creating training job '$1

training_image="117617958416.dkr.ecr.eu-west-1.amazonaws.com/robocars:latest"
iam_role_arn="arn:aws:iam::117617958416:role/robocar-training"
DATA_BUCKET="s3://robocars-cyrilix-learning/input"
DATA_OUTPUT="s3://robocars-cyrilix-learning/output"

aws sagemaker create-training-job \
    --training-job-name ${job_name} \
    --hyper-parameters '{ "sagemaker_region": "\"eu-west-1\"", "with_slide": "true" }' \
    --algorithm-specification TrainingImage="${training_image}",TrainingInputMode=File \
    --role-arn ${iam_role_arn} \
    --input-data-config "[{ \"ChannelName\": \"train\", \"DataSource\": { \"S3DataSource\": { \"S3DataType\": \"S3Prefix\", \"S3Uri\": \"${DATA_BUCKET}\", \"S3DataDistributionType\": \"FullyReplicated\" }} }]" \
    --output-data-config S3OutputPath=${DATA_OUTPUT} \
    --resource-config InstanceType=ml.p2.xlarge,InstanceCount=1,VolumeSizeInGB=1 \
    --stopping-condition MaxRuntimeInSeconds=1800

FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY main.py .

COPY processed_data/ ./processed_data/

CMD ["main.handler"]
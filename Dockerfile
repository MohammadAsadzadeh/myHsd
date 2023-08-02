FROM python:3.9
WORKDIR /code
RUN pip install --upgrade pip
# COPY requirements.txt /code/
# RUN pip install -r requirements.txt



RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install openpyxl
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install hazm
RUN pip install python-bidi
RUN pip install arabic-reshaper
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install nvidia-cudnn-cu11
RUN pip install torch
RUN pip install transformers
RUN pip install gdown
Run pip install jdatetime
RUN mkdir /code/app
# COPY ./ /code/app
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
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
RUN pip install jdatetime
RUN pip install mysql-connector-python
RUN pip install pyjwt
RUN pip install bcrypt
RUN mkdir /code/app
ENV DB_PASS=123
ENV ACCESS_TOKEN=token
ENV ADMIN_USERNAME=admin
ENV ADMIN_PASSWORD=admin
ENV SECRET_KEY=your_secret_key
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
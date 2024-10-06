FROM python:3.11.1-buster

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY GemmArte/ GemmArte
COPY paligemma-3b-pt-224/ paligemma-3b-pt-224
COPY pipeline.py .
COPY rp_handler.py .

CMD [ "python", "-u" ,"./rp_handler.py" ]
FROM python:3.11.5-bookworm

RUN pip install --upgrade pip

WORKDIR /root/source_code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# COPY ./source_code /root/source_code

CMD tail -f /dev/null
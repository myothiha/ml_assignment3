FROM python:3.11.5-bookworm

RUN pip install --upgrade pip

WORKDIR /root/source_code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# CMD tail -f /dev/null
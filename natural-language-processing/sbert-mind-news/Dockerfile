FROM chromadb/chroma

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python3", "-m" , "flask", "--app", "app.py", "--debug", "run", "--host=0.0.0.0"]
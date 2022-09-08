FROM tensorflow/tensorflow

COPY city_categorization /city_categorization
COPY requirements.txt /requirements.txt
COPY model /model

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn city_categorization.api.fast:app --host 0.0.0.0

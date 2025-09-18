FROM python:3.10
ADD Datagen.py .
ADD sample_logins.csv .
RUN pip install numpy pandas matplotlib scikit-learn joblib 

CMD ["python", "./Datagen.py"] 




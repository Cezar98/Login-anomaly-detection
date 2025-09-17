FROM python:3.10
ADD Datagen.py .
RUN pip install numpy pandas matplotlib scikit-learn joblib python-dotenv
CMD [“python”, “./Datagen.py”] 
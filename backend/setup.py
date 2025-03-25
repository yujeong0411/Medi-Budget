# pip install -e . 실행

from setuptools import setup, find_packages

setup(
    name='medi-budget',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "numpy==1.23.0",
        "matplotlib==3.9.4",
        "psycopg2-binary==2.9.10",
        "sqlalchemy==2.0.39",
        "python-dotenv==0.19.1",
        "fastapi",
        "uvicorn",
        "pydantic",
        "scikit-learn==1.6.1",
        "scipy==1.10.1",
        "seaborn==0.13.2",
        "openpyxl==3.1.5",
        "beautifulsoup4==4.13.3",
        "requests==2.32.3",
        "playwright==1.50.0",
        "pyee==12.1.1",
        "xgboost==2.1.4"
    ]
)
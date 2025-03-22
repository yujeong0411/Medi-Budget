from sqlalchemy import Column, Integer, String, Text 
from pydantic import BaseModel

class Disease(BaseModel):
    disease_id: int
    disease_name: str
    disease_code: str

class Hospital_type(BaseModel):
    hospital_type_id: int
    hospital_type: str

class Region(BaseModel):
    region_id: int
    region_name: str


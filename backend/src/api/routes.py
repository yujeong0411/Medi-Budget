from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from typing import List, Dict, Any
import pandas as pd

from configs.db_config import get_db_engine

router = APIRouter()

# 질병 조회 
@router.get("/diseases", response_model=List[Dict[str, Any]])
async def get_diseases():
    try:
        engine = get_db_engine()
        query = "SELECT disease_id, disease_name, disease_code FROM diseases ORDER BY disease_name"
        df = pd.read_sql_query(query, engine)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질병 목록 조회 오류: {str(e)}")
    
# 의료기관 종별 조회
@router.get("/hospital-types", response_model=List[Dict[str, Any]])
async def get_hospital_types():
    try:
        engine = get_db_engine()
        query = "SELECT hospital_type_id, hospital_type FROM hospital_types ORDER BY hospital_type_id"
        df = pd.read_sql_query(query, engine)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"병원 목록 조회 오류: {str(e)}")
    

# 지역 조회
@router.get("/regions", response_model=List[Dict[str, Any]])
async def get_regions():
    try:
        engine = get_db_engine()
        query = "SELECT region_id, region_name FROM regions ORDER BY region_name"
        df = pd.read_sql_query(query, engine)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지역 목록 조회 오류: {str(e)}")
    


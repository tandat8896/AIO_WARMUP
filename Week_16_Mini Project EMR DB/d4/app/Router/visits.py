from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from app.database.database import execute_query
from app.models.schemas import Visit

router = APIRouter()

@router.get("/visits/", response_model=List[Visit])
def get_visits(
    patient_id: str = None,
    doctor_id: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    skip: int = 0,
    limit: int = 100
):
    query = "SELECT * FROM emr_visits WHERE 1=1"
    params = []
    
    if patient_id:
        query += " AND patient_id = %s"
        params.append(patient_id)
    if doctor_id:
        query += " AND doctor_id = %s"
        params.append(doctor_id)
    if start_date:
        query += " AND visit_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND visit_date <= %s"
        params.append(end_date)
    
    query += " LIMIT %s OFFSET %s"
    params.extend([limit, skip])
    
    visits = execute_query(query, tuple(params))
    if visits is None:
        raise HTTPException(status_code=500, detail="Database error")
    return visits

@router.get("/visits/{visit_id}", response_model=Visit)
def get_visit(visit_id: str):
    query = """
    SELECT * FROM emr_visits 
    WHERE visit_id = %s
    """
    visit = execute_query(query, (visit_id,))
    if visit is None:
        raise HTTPException(status_code=500, detail="Database error")
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")
    return visit[0] 
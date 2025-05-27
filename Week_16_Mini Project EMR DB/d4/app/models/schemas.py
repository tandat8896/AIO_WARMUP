from pydantic import BaseModel
from typing import List,Optional
from datetime import datetime

class Patient(BaseModel):
    patient_id: str
    medical_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    address: str
    phone_number: str
    email: str
    created_at: str
    updated_at: str

class Visit(BaseModel):
    visit_id: str
    patient_id: str
    doctor_id: str
    department_id: str
    facility_id: str
    visit_date: str
    symptoms: str
    diagnosis: str
    notes: str
    status: str
    created_at: str
    updated_at: str

class Test(BaseModel):
    test_id: str
    visit_id: str
    test_name: str
    test_type: str
    ordered_by: str
    ordered_date: str
    performed_date: str
    results: str
    status: str
    created_at: str
    updated_at: str

class Prescription(BaseModel):
    prescription_id: str
    visit_id: str
    medication_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str
    prescribed_by: str
    prescribed_date: str
    status: str
    created_at: str
    updated_at: str

class Test(BaseModel):
    test_id: str
    visit_id: str
    test_name: str
    test_date: datetime
    results: str
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class VisitDetail(Visit):
    prescriptions: List[Prescription] = []
    tests: List[Test] = []

class PatientHistory(Patient):
    visits: List[VisitDetail] = [] 
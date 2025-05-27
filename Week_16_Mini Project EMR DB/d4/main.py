from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Create FastAPI app
app = FastAPI(
    title="EMR System API",
    description="Electronic Medical Record System API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic health check endpoint
@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "message": "Server is running"}

# Import and include routers
from app.Router import patients, visits, medical_history

app.include_router(patients.router, prefix="/api/v1", tags=["patients"])
app.include_router(visits.router, prefix="/api/v1", tags=["visits"])
app.include_router(medical_history.router, prefix="/api/v1", tags=["medical_history"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

